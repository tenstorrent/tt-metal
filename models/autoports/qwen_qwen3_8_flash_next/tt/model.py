# SPDX-License-Identifier: Apache-2.0
"""Qwen Flash-Next text model on the qualified TP4 decoder contract.

The residual remains four local 640-wide HC streams between every layer.
Embeddings are hidden-sharded; only the terminal mixed vector is gathered.
The untied LM head is vocabulary-sharded. No runtime host tensor conversion.
"""

import json
import math
from dataclasses import dataclass

import ttnn

from .checkpoint import CheckpointState
from .multichip_decoder import MeshOps, MultichipDecoder, _hidden_parts


@dataclass
class ModelCache:
    """Explicit external ownership of every attention and PLE request buffer."""

    layers: list[dict]


def _share_serving_constants(layer, pool, rope_parameters):
    """Deduplicate immutable setup constants before any trace is prepared.

    GDN constructs these sources with torch.zeros; reset_state_inplace copies
    FROM them and never mutates them. QSA cos/sin depend only on context and the
    shared default-RoPE config and are read-only embedding tables. Request
    banks, mutable workspaces, and weights are not shared by this helper.
    """
    attention = layer.attention
    if layer.kind == "linear_attention":
        names, semantics = ("zero_recurrent", "zero_conv"), ("zeros",)
    else:
        names = ("cos", "sin")
        semantics = ("qsa_rope64", attention.max_context, json.dumps(rope_parameters, sort_keys=True))
    for name in names:
        tensor = getattr(attention, name)
        candidates = pool.setdefault((name, semantics), [])
        for owner_mesh, shared in candidates:
            if (
                owner_mesh is layer.mesh
                and tuple(shared.shape) == tuple(tensor.shape)
                and tuple(shared.padded_shape) == tuple(tensor.padded_shape)
                and shared.dtype == tensor.dtype
                and shared.layout == tensor.layout
                and shared.memory_config() == tensor.memory_config()
            ):
                # Release only this layer's redundant freshly constructed
                # constant; all consumers retain the shared backing allocation.
                setattr(attention, name, shared)
                break
        else:
            candidates.append((layer.mesh, tensor))


def _share_serving_moe_capacity(layer, pool):
    """Share singleton dispatch scratch whose live rows are rewritten before reads.

    Serving executes every native op and trace on CQ0. Each lane dispatches
    row0 for every active expert and defines all128 counts before the FFN reads
    it; inactive experts and padded rows are not read. ROW_MAJOR BF16 FFN output
    is a separate tensor, so no returned output aliases this scratch. The next
    lane/layer/trace writes it only after the previous CQ0 reader has completed.
    All sharing finishes before generator-wide warmup/capture.
    """
    tensor = layer.mlp.capacity
    if (
        tuple(tensor.shape) != (1, 1, 128 * 32, 2560)
        or tensor.dtype != ttnn.bfloat16
        or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        return
    candidates = pool.setdefault("moe_singleton_dispatch", [])
    for owner_mesh, shared in candidates:
        if (
            owner_mesh is layer.mesh
            and tuple(shared.padded_shape) == tuple(tensor.padded_shape)
            and shared.memory_config() == tensor.memory_config()
        ):
            layer.mlp.capacity = shared
            return
    candidates.append((layer.mesh, tensor))


def _share_serving_gdn_work(layer, pool):
    """Reuse slotted GDN work after each layer scatters to its own state bank.

    Every slotted invocation gathers ALL work tiles before consuming them,
    including zero initialization at position0, and stores them before return.
    CQ0 serializes the next layer's gather after this scatter. Attention outputs
    do not alias these work tensors. Fixed-lane continuation is incompatible;
    serving model.forward therefore requires explicit slots. Banks stay private.
    """
    if layer.kind != "linear_attention":
        return
    attention = layer.attention
    for name in ("recurrent_state", "conv_state"):
        tensor = getattr(attention, name)
        candidates = pool.setdefault(("slotted_gdn_work", name), [])
        for owner_mesh, shared in candidates:
            if (
                owner_mesh is layer.mesh
                and tuple(shared.shape) == tuple(tensor.shape)
                and tuple(shared.padded_shape) == tuple(tensor.padded_shape)
                and shared.dtype == tensor.dtype
                and shared.layout == tensor.layout
                and shared.memory_config() == tensor.memory_config()
            ):
                setattr(attention, name, shared)
                break
        else:
            candidates.append((layer.mesh, tensor))


def _share_serving_qsa_work(layer, pool):
    """Reuse paged QSA's one-lane dense scoring workspace.

    The gather writes the completed pooled prefix and zeros its partial final
    tile. Whole suffix tiles remain unspecified; the score mask overwrites their
    scores before selection, including scores produced from stale NaNs. The
    score reader completes on CQ0 before the next lane/layer gather. Selected
    IDs and returned attention outputs use separate storage. Physical raw,
    packed and pooled caches remain private and caller-owned.
    """
    attention = layer.attention
    if layer.kind == "linear_attention" or not attention.paged_pooled:
        return
    tensor = attention.pooled_keys
    if (
        tuple(tensor.shape) != (1, 1, attention.blocks, 128)
        or tensor.dtype != ttnn.bfloat16
        or tensor.layout != ttnn.TILE_LAYOUT
    ):
        return
    candidates = pool.setdefault("paged_qsa_scoring_work", [])
    for owner_mesh, shared in candidates:
        if (
            owner_mesh is layer.mesh
            and tuple(shared.shape) == tuple(tensor.shape)
            and tuple(shared.padded_shape) == tuple(tensor.padded_shape)
            and shared.dtype == tensor.dtype
            and shared.layout == tensor.layout
            and shared.memory_config() == tensor.memory_config()
        ):
            attention.pooled_keys = shared
            return
    candidates.append((layer.mesh, tensor))


class TerminalMixer:
    def __init__(self, state, cfg, ops):
        self.ops, self.eps = ops, cfg.rms_norm_eps
        prefix = "model.language_model.hyper_connection_mixer."
        self.norm = ops.shards(
            [x.reshape(1, 1, 4, 640) for x in _hidden_parts(1 + state[prefix + "hc_norm.weight"].float())]
        )
        self.down = ops.shards(
            [x.T for x in _hidden_parts(state[prefix + "input_mix_weight_down.weight"] / cfg.hc_count)], ttnn.bfloat16
        )
        self.up = ops.shards(_hidden_parts(state[prefix + "input_mix_weight_up.weight"].T), ttnn.bfloat16)

    def __call__(self, x):
        b, s = tuple(x.shape)[1:3]
        streams = self.ops.norm(ttnn.reshape(x, (1, b * s, 4, 640)), self.norm, self.eps)
        normalized = ttnn.reshape(streams, (1, b, s, 2560))
        low = self.ops.reduce(
            self.ops.linear(normalized, self.down, compute_kernel_config=self.ops.compute, dtype=ttnn.float32)
        )
        mix = ttnn.sigmoid(
            self.ops.linear(
                ttnn.silu(ttnn.typecast(low, ttnn.bfloat16)), self.up, compute_kernel_config=self.ops.compute
            )
        )
        mixed = ttnn.mean(ttnn.multiply(ttnn.reshape(mix, (1, b * s, 4, 640)), streams), dim=2, keepdim=True)
        return self.ops.gather(ttnn.reshape(mixed, (1, b, s, 640)))


class QwenFlashNextModel:
    def __init__(
        self,
        checkpoint,
        mesh_device,
        *,
        batch_size=1,
        max_requests=None,
        layer_indices=None,
        max_seq_len=None,
        cache_dir=None,
        precision_config=None,
        attention_pool_tokens=None,
    ):
        import copy

        from transformers import GenerationConfig
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

        from .precision import apply_layer, cast_attr, compute, for_layer, load_policy

        self.precision_policy = load_policy(precision_config)
        self.residual_dtype = getattr(ttnn, self.precision_policy["residual_dtype"])
        self.logits_dtype = getattr(ttnn, self.precision_policy["logits_dtype"])
        self.mesh = mesh_device
        self.ops = MeshOps(mesh_device)
        self.ops.precision_ccl = self.precision_policy["ccl_dtype"]
        # vLLM registers its own Qwen config globally. Preserve the same explicit
        # HF text architecture used to qualify the selected precision policy.
        self.cfg = Qwen4ExpConfig.from_pretrained(checkpoint, local_files_only=True).text_config
        self.generation_config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
        if self.cfg.max_position_embeddings != self.precision_policy["max_context"]:
            raise ValueError("Precision context contract disagrees with checkpoint")
        self.supported_context = self.precision_policy["max_context"]
        self.context = max_seq_len or self.supported_context
        if not 2048 <= self.context <= self.supported_context or self.context % 128:
            raise ValueError("Physical cache capacity must be a multiple of128 within [2048,262144]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an integer within [1,32]")
        self.batch_size = batch_size
        self.max_requests = batch_size if max_requests is None else max_requests
        if not isinstance(self.max_requests, int) or self.max_requests < batch_size:
            raise ValueError("max_requests must cover all active batch rows")
        # Caller-owned request slots precede a private scratch slot per lane.
        self.state_slots = self.max_requests + batch_size
        self.serving_pool = attention_pool_tokens is not None
        self.public_pool_tokens = batch_size * self.context if attention_pool_tokens is None else attention_pool_tokens
        if not isinstance(self.public_pool_tokens, int) or self.public_pool_tokens < 32 or self.public_pool_tokens % 32:
            raise ValueError("Attention pool tokens must be a positive multiple of32")
        self.scratch_page_base = self.public_pool_tokens // 32
        self.layer_indices = list(range(self.cfg.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        state = CheckpointState(checkpoint)
        emb = state["model.language_model.embed_tokens.weight"]
        self.embedding = self.ops.shards(
            [emb[:, r * 640 : (r + 1) * 640] for r in range(4)], layout=ttnn.ROW_MAJOR_LAYOUT
        )
        head = emb if self.cfg.tie_word_embeddings else state["lm_head.weight"]
        self.lm_head = self.ops.shards([part.T.contiguous() for part in head.chunk(4, dim=0)])
        cast_attr(self, "lm_head", self.precision_policy["weight_groups"]["lm_head"])
        self.head_compute = compute(mesh_device, self.precision_policy["compute_fidelities"]["lm_head"])
        del emb, head
        self.terminal = TerminalMixer(state, self.cfg, self.ops)
        cfg = copy.deepcopy(self.cfg)
        cfg.max_position_embeddings = self.context
        self.layers = []
        serving_constants = {}
        for idx in self.layer_indices:
            print("LOAD_LAYER", idx, flush=True)
            layer = MultichipDecoder.from_state_dict(
                state,
                hf_config=cfg,
                layer_idx=idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=self.state_slots,
                chunk_size=512 if batch_size == 1 else 128,
                cache_dir=cache_dir,
                attention_pool_tokens=(self.public_pool_tokens + batch_size * 512) if self.serving_pool else None,
            )
            apply_layer(layer, for_layer(self.precision_policy, idx))
            if self.serving_pool:
                _share_serving_constants(layer, serving_constants, cfg.rope_parameters)
                _share_serving_moe_capacity(layer, serving_constants)
                _share_serving_gdn_work(layer, serving_constants)
                _share_serving_qsa_work(layer, serving_constants)
            if layer.kind != "linear_attention" and not self.serving_pool:
                # Dedicated physical scratch pages let inactive fixed rows execute
                # without writing any live request's KV or raw-index pages.
                attention = layer.attention
                attention.pool_tokens += batch_size * 512
                del attention.packed_cache, attention.raw_key_cache
                attention.packed_cache = ttnn.empty(
                    (1, 1, attention.pool_tokens, 512),
                    dtype=attention.cache_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                )
                attention.raw_key_cache = ttnn.empty(
                    (attention.pool_tokens // 32, 1, 32, 128),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                )
            self.layers.append(layer)
        self.cache = self.export_cache()
        self._bound_cache = self.cache

    def export_cache(self):
        result = []
        for layer in self.layers:
            names = (
                ("recurrent_state", "recurrent_bank", "conv_state", "conv_bank")
                if layer.kind == "linear_attention"
                else (
                    ("packed_cache", "raw_key_cache", "pooled_cache", "pooled_keys")
                    if self.serving_pool
                    else ("packed_cache", "raw_key_cache", "pooled_keys", "pooled_bank")
                )
            )
            row = {name: getattr(layer.attention, name) for name in names}
            if layer.ple is not None:
                row["ple_cache"] = layer.ple_cache
            result.append(row)
        return ModelCache(result)

    def bind_cache(self, cache):
        if cache is self._bound_cache:
            return
        if not isinstance(cache, ModelCache) or len(cache.layers) != len(self.layers):
            raise ValueError("Expected a ModelCache with one state mapping per loaded layer")
        for supplied, expected in zip(cache.layers, self.cache.layers):
            if supplied.keys() != expected.keys():
                raise ValueError("External cache must supply every decoder state buffer")
            for name, tensor in supplied.items():
                if name == "ple_cache" and vars(tensor).keys() != vars(expected[name]).keys():
                    raise ValueError("External PLE cache must supply token and convolution history")
                pairs = (
                    zip(vars(tensor).values(), vars(expected[name]).values())
                    if name == "ple_cache"
                    else [(tensor, expected[name])]
                )
                for actual, reference in pairs:
                    if (tuple(actual.shape), actual.dtype, actual.layout, actual.memory_config()) != (
                        tuple(reference.shape),
                        reference.dtype,
                        reference.layout,
                        reference.memory_config(),
                    ):
                        raise ValueError(
                            f"External cache {name} must preserve shape, dtype, layout and memory placement"
                        )
        for layer, state in zip(self.layers, cache.layers):
            for name, tensor in state.items():
                if name == "ple_cache":
                    layer.ple_cache = tensor
                else:
                    setattr(layer.attention, name, tensor)
        # Transfer the owned reference too; do not retain an unused cache pool.
        self.cache = cache
        self._bound_cache = cache

    def reset_cache(self, cache):
        from .state_ops import zero_inplace

        for state in cache.layers:
            for name, tensor in state.items():
                if name == "ple_cache":
                    for value in vars(tensor).values():
                        zero_inplace(self.mesh, value)
                else:
                    zero_inplace(self.mesh, tensor)

    def prepare(self, lengths):
        for layer in self.layers:
            layer.prepare(lengths)

    def embed(self, tokens):
        b, s = tokens.shape
        x = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, b, s, 640))
        residual = ttnn.concat([x] * 4, dim=-1)
        assert residual.dtype == self.residual_dtype
        return residual

    def forward(self, tokens, *, positions, slots, page_table, kv_cache, decode=False, last_token_only=False):
        if self.serving_pool and slots is None:
            raise ValueError("Serving GDN work sharing requires explicit request slots")
        self.bind_cache(kv_cache)
        x = self.embed(tokens)
        for layer in self.layers:
            fn = layer.decode_forward if decode else layer.prefill_forward
            x = fn(x, current_positions=positions, slots=slots, tokens=tokens, page_table=page_table)
            assert x.dtype == self.residual_dtype
        mixed = self.terminal(x)
        if last_token_only:
            # Preserve the terminal mixer's sequence-dependent norm policy.
            # Cache and mixer still consume every real token; only the costly
            # vocabulary projection is restricted to the last logical row.
            mixed = mixed[:, :, -1:, :]
        # Vocabulary width needs more cores and bounded K buffers than the
        # decoder's narrow projections. Precision comes from the selected policy.
        rows = self.batch_size * mixed.shape[-2]
        if decode and self.batch_size > 1 and mixed.shape[-2] == 1:
            # Compact logical decode lanes before the vocabulary projection.
            # [1,B,1,K] pads every lane to 32 M rows; ROW_MAJOR removes those
            # gaps before forming [1,1,B,K]. The public decode logits already
            # use this compact row order. Keep B1 and prefill paths unchanged.
            mixed = ttnn.to_layout(mixed, ttnn.ROW_MAJOR_LAYOUT)
            mixed = ttnn.reshape(mixed, (1, 1, self.batch_size, mixed.shape[-1]))
            mixed = ttnn.to_layout(mixed, ttnn.TILE_LAYOUT)
        if rows == 1:
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(11, 10),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=math.ceil((self.cfg.vocab_size // 4) // 32 / 110),
                fuse_batch=True,
                mcast_in0=True,
            )
            logits = ttnn.linear(
                ttnn.to_memory_config(mixed, ttnn.L1_MEMORY_CONFIG),
                self.lm_head,
                program_config=program,
                compute_kernel_config=self.head_compute,
                dtype=self.logits_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            logits = ttnn.linear(
                mixed,
                self.lm_head,
                compute_kernel_config=self.head_compute,
                dtype=self.logits_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.reshape(logits, (1, 1, self.batch_size, self.cfg.vocab_size // 4)) if decode else logits
