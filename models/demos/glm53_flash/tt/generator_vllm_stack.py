# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash on the persistent-stack port (models/autoports/zai_org_glm_5_3_flash/tt: ``StackBackend`` under
``Glm53Generator``) for the TT vLLM plugin -- the FIRST adapter: text only, greedy on device (the head's top-1 inside
the one full-width decode trace), host-sampled rows through full logits, no prefix cache, no multimodal, no speculative
decode, synchronous scheduling (every decode reloads its inputs).

Selected with ``TT_MODEL_CLASS_OVERRIDES="Glm5NextForCausalLM=<ADAPTER_TARGET>,Glm5NextForConditionalGeneration=
<ADAPTER_TARGET>"``. The plugin opens the mesh BEFORE ``initialize_vllm_model`` (worker.py:274-281) with
``trace_region_size`` / ``worker_l1_size`` / ``l1_small_size`` from the launch's ``tt`` config (worker.py:1026-1041)
and the fabric from ``model_capabilities["fabric_config"]`` (worker.py:994-1010), so the validated 8x4 launch needs
``trace_region_size = galaxy_defaults.trace_mb(max_model_len) << 20``, ``worker_l1_size 1280896``, ``l1_small_size
32768`` and ``--block-size 64`` on the command line; ``galaxy_defaults.apply`` sets the remaining GLM53_* knobs at
model construction (its GLM53_MODEL_TRACE_MB is dead here: the device is already open).

Slot aliasing: the plugin's ``empty_slots`` / ``slot_remap`` name PLUGIN slots, which the adapter maps onto the
generator's slots (``_CacheBinding.slots``) through a bijection it keeps (``_alias``). The plugin's decode gather
(model_runner.py:1366-1421: a full permutation of every slot after it condenses the rows) is absorbed into the alias
instead of moving state: row ``i`` decodes in the generator slot its request was admitted to
(``Glm53Generator.decode_forward(slots=...)``), so no ``remap_slots`` copies run and a gather that is a cycle over a
full batch (every slot live: the generator's "cyclic slot_remap needs a free scratch slot" refusal, hit by the first
4-user served step) is fine. A new request takes the lowest free generator slot. vLLM's logical blocks travel
``pt_shadow -> OwnerBlockMap -> per-chip local tables`` inside the generator / backend once the cache is allocated
with ``null_block=0``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
    BLOCK_SIZE,
    MAX_DECODE_SLOTS,
    DecodeOutput,
    PendingTokens,
)
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams

HF_ARCHITECTURES = ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
ADAPTER_TARGET = "models.demos.glm53_flash.tt.generator_vllm_stack:GLM53StackForCausalLM"
LATENT_DIM = 512  # kv_lora_rank: the latent page row, the only "KV" vLLM accounts for (generator_vllm.LATENT_DIM)


def _mesh_shape_from_env() -> tuple[int, int]:
    """The mesh the plugin will open (``generator_vllm._mesh_shape_from_env``): ``GLM53_MESH_SHAPE`` (rows,cols) if
    set, else the plugin's ``MESH_DEVICE`` presets this model runs on or a ``(r, c)`` tuple."""
    spec = os.environ.get("GLM53_MESH_SHAPE")
    if spec:
        r, c = (int(v) for v in spec.split(","))
        return r, c
    md = os.environ.get("MESH_DEVICE", "")
    if md.startswith("("):
        r, c = (int(v) for v in md.strip("() ").split(","))
        return r, c
    return {"BH-Galaxy": (8, 4), "TG": (8, 4), "P150x8": (1, 8), "P150x4": (1, 4)}.get(md, (8, 4))


def _fabric_capability() -> dict | None:
    """``model_capabilities["fabric_config"]`` (MODEL_CAPABILITIES.md:126; merged into ``ttnn.set_fabric_config(**kw)``
    by worker.py:994-1010): ``config`` from ``stack_model.serving_fabric_config`` (FABRIC_1D, the fabric every stack
    row was validated with -- the EP all-to-all and the sharded pool's streams, GLM53_FABRIC=1d) plus the relaxed-init
    ``reliability_mode`` and the maximum-payload ``router_config`` of ``tt/mesh.py::mesh_device_params`` (what the e2e
    test's device fixture set), the keys ``generator_vllm._fabric_capability`` declares. Built only in a process that
    already imported ttnn (the plugin does before it resolves the model class): importing this module stays ttnn-free.
    """
    if "ttnn" not in sys.modules:
        if "vllm" not in sys.modules:  # a host process (the host test keeps this module ttnn-free)
            return None
        import ttnn  # noqa: F401  (a vLLM process: the validated fabric must reach the plugin before it opens the mesh)
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import serving_fabric_config
        from models.demos.glm53_flash.tt.mesh import mesh_device_params

        out = dict(serving_fabric_config())
        params = mesh_device_params(*_mesh_shape_from_env())
    except Exception as exc:  # pragma: no cover - device library detail
        logger.warning(
            f"GLM53 stack vLLM: fabric capability not resolved ({exc}); the plugin's hardware default applies"
        )
        return None
    if "reliability_mode" in params:
        out["reliability_mode"] = params["reliability_mode"]
    if "fabric_router_config" in params:
        out["router_config"] = params["fabric_router_config"]
    return out


def _snapshot_dir(hf_config, default_snapshot_dir) -> Path:
    """The checkpoint directory: the HF config's ``_name_or_path`` when it is a snapshot, else the default."""
    name = getattr(hf_config, "_name_or_path", "") or ""
    p = Path(name)
    if name and p.is_dir() and (p / "config.json").exists():
        return p
    return default_snapshot_dir()


def trim_page_rows(table) -> torch.Tensor:
    """The plugin's block table ``[B, blocks]`` (input_batch.py:789-808: zero past each row's live blocks; 0 is vLLM's
    null block, never allocated) -> the generator's negative-terminated rows: every column from the first zero at
    column >= 1 on becomes -1. ``Glm53Generator._write_page_rows`` cuts a row at its first negative entry; a zero
    there would be the null block in a covered column (``allocate_kv_cache(null_block=0)`` refuses it) and would also
    defeat the unchanged-row fast path (``_rows_unchanged`` wants a negative right after the coverage). Column 0 is
    kept as given: a live row never holds the null block there, and a dead row (``start_pos`` -1) is never read."""
    t = torch.as_tensor(table).to(torch.int64).clone()
    if t.dim() == 1:
        t = t[None]
    if t.shape[1] > 1:
        pad = (t[:, 1:] == 0).cumsum(dim=1) > 0
        t[:, 1:] = torch.where(pad, torch.full_like(t[:, 1:], -1), t[:, 1:])
    return t


def row_params(sampling_params, r: int) -> SamplingParams:
    """Row ``r`` of the plugin's per-row lists (``TTSamplingParams``, model_input.py:69-84; on device-sampling calls
    the fields are lists, a None seed = vLLM gave none) -> the generator's ``SamplingParams``: temperature 0 or top_k 1
    is greedy (``is_greedy``: the head's argmax, the only device route of this adapter); vLLM's "no top-k" (0 / -1)
    becomes None. A missing field reads as greedy / no top-k / top-p 1 / no seed."""

    def val(name, default):
        v = getattr(sampling_params, name, None)
        if v is None or r >= len(v) or v[r] is None:
            return default
        return v[r]

    t = float(val("temperature", 0.0))
    k = int(val("top_k", 1))
    seed = val("seed", None)
    return SamplingParams(
        temperature=t,
        top_k=None if k <= 0 else k,
        top_p=float(val("top_p", 1.0)),
        seed=None if seed is None or int(seed) < 0 else int(seed),
    )


def _is_host_output(out) -> bool:
    """async_decode.py:126-133 ``_is_host_decode_output``: a tensor or a tuple of tensors / None needs no read."""
    return isinstance(out, torch.Tensor) or (
        isinstance(out, tuple) and all(t is None or isinstance(t, torch.Tensor) for t in out)
    )


class GLM53StackForCausalLM:
    """GLM-5.3-Flash text generation on the persistent stack (one Blackhole Galaxy mesh) for the TT vLLM plugin."""

    # DECODE_RELOAD_CONTRACT.md / async_decode.py:1398-1412: the four explicit reload commands are accepted
    decode_input_update_contract = 1

    model_capabilities = {
        # MODEL_CAPABILITIES.md:56-59 scheduling flags: a prompt split across engine steps resumes through the
        # generator's continuation chunks (64-aligned, 128 with the TP prefill's SDPA: ``continuation_align``)
        "supports_chunked_prefill": True,
        "supports_prefix_caching": False,
        "confirms_prefix_cache_hits": False,
        # MODEL_CAPABILITIES.md:85-91 sampling / async flags: the head's argmax is the only device sampler
        # (``StackBackend(topk=1, sample=False)``), so greedy rows (and top_k == 1) decode on device and every other
        # sampled row goes to the host sampler over full logits; no resident-input chain (every decode reloads)
        "supports_sample_on_device": True,
        "max_device_top_k": 1,
        "device_top_k_unbounded_as": None,
        "supports_device_penalties": False,
        "supports_device_grammar": False,
        "supports_async_decode": False,
        "supports_async_spec_decode": False,
        "supports_compact_host_logits": False,
        "supports_selective_host_readback": False,
        # MODEL_CAPABILITIES.md:126 device setup: worker.py:274-281 reads the class attribute before it opens the mesh
        "fabric_config": _fabric_capability(),
        # SPEC_DECODE_CONTRACT.md: the stack's speculative step serves one live user in slot 0 only -- off
        "supports_spec_decode": False,
    }

    # model_runner.py:2301-2320: the one captured decode width (fixed user slots, no smaller bucket); set by
    # ``warmup_model_decode(enable_trace=True)``, cleared by ``release_persistent_capture``
    tt_supported_decode_batch_sizes: tuple[int, ...] = ()
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False  # worker.py:799-807: one logical KV group
    supports_multimodal = False  # vLLM's ``SupportsMultiModal`` flag: text only

    def __init__(self, generator, max_batch_size: int, max_seq_len: int, *, source=None, **vllm_kwargs):
        # ``**vllm_kwargs`` (``vllm_config`` / ``prefix``) is never used: the plugin constructs the adapter through
        # ``initialize_vllm_model``; vLLM >= 0.30's ``is_vllm_model`` wants the keyword accepted (see the shims below)
        self.gen = generator
        self.max_batch = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.source = source
        self.vocab_size = int(generator.vocab_size)
        self.kv_cache = None
        self._alias = list(range(self.max_batch))  # plugin slot -> generator slot (identity until the first gather)

    # ================================================================================= construction
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
    ):
        """loader.py:38-45: ``initialize_vllm_model(hf_config, device, max_batch_size, max_seq_len=max_model_len,
        tt_data_parallel=..., optimizations=...)`` on the already-open mesh. ``max_num_seqs`` is the backend's
        construction ``max_batch`` (<= 32; the trace sizes, sparse ladder and buckets are construction-time: another
        ``max_model_len`` is a rebuild); ``tt_data_parallel`` must be 1 (one engine per mesh). The validated 8x4
        knobs come from ``galaxy_defaults.apply`` (an explicit environment value always wins), the weights from
        ``mk_sources.install_weight_source`` (GLM53_MK_TILE_CACHE_V3, built on a cold cache), the backend as the e2e
        test builds it (topk 1, sample False, EP experts, native_t above 4 users). ``optimizations`` is ignored."""
        if int(tt_data_parallel) != 1:
            raise ValueError("GLM53StackForCausalLM runs one engine per mesh (tt_data_parallel must be 1)")
        B, seq = int(max_batch_size), int(max_seq_len)
        if not 1 <= B <= MAX_DECODE_SLOTS:
            raise ValueError(f"max_num_seqs {B} outside [1, {MAX_DECODE_SLOTS}] (the stack's decode slots)")
        from models.autoports.zai_org_glm_5_3_flash.tt import galaxy_defaults as GD
        from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_sources import MkTileCacheSource, install_weight_source
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend
        from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer
        from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir

        snapshot = _snapshot_dir(hf_config, default_snapshot_dir)
        shape = tuple(int(v) for v in list(mesh_device.shape))
        applied = GD.apply(shape, seq=seq)  # (its prefill buckets are the ones that fit `seq`)
        logger.info(f"GLM53 stack vLLM: mesh {shape}, max_num_seqs {B}, max_model_len {seq}, galaxy defaults {applied}")
        cfg = load_hf_text_config(str(snapshot))
        source = install_weight_source(snapshot, cfg)
        layers = list(range(int(n_layers) if n_layers else int(cfg.num_hidden_layers)))
        # the narrow widths (GLM53_STACK_NARROW_WIDTHS) reuse the cached layer weights: an empty dict this build fills
        narrow = os.environ.get("GLM53_STACK_NARROW_WIDTHS", "none") not in ("", "none")
        t0 = time.perf_counter()
        backend = StackBackend(
            mesh_device,
            max_batch=B,
            layers=layers,
            hf_config=cfg,
            topk=1,
            sample=False,
            ep_weights=True,
            native_t=B > 4,
            weight_cache={} if narrow else None,
        )
        if isinstance(source, MkTileCacheSource) and not source.ready:
            source.complete()  # built by this construction: a later boot reads every tensor from the cache
        logger.info(f"GLM53 stack vLLM: StackBackend built in {time.perf_counter() - t0:.0f} s ({len(layers)} layers)")
        gen = Glm53Generator(backend, tokenizer=load_tokenizer(snapshot), max_batch=B, max_seq_len=seq)
        return cls(gen, B, seq, source=source)

    # ================================================================================= KV cache
    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """worker.py:737-785: the all-user token budget vLLM sizes its block pool from (plus one block per user on
        top): every slot may hold ``max_model_len`` latent pages, so ``max_num_seqs x max_model_len``."""
        if max_model_len is None or max_num_seqs is None:
            raise ValueError("get_max_tokens_all_users needs max_model_len and max_num_seqs")
        return int(max_num_seqs) * int(max_model_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """worker.py:349-416 / 630-652: one logical full-attention group of latent pages, ``[64, 1, 512]`` bf16 per
        block, so vLLM's admission accounting counts what the stack pages; the device cache is model-owned. The block
        size must be the stack's 64-token page (``--block-size 64``)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        block_size = int(vllm_config.cache_config.block_size)
        if block_size != BLOCK_SIZE:
            raise ValueError(f"--block-size {block_size} must be {BLOCK_SIZE}: the stack's latent page is 64 tokens")
        return {
            "glm53_latent": FullAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=LATENT_DIM, dtype=torch.bfloat16
            )
        }

    def allocate_kv_cache(self, shape, dtype, num_layers: int):
        """model_runner.py:786-816: ``allocate_kv_cache(shape=(num_blocks, heads, block_size, head), dtype,
        num_layers)``; the return value comes back as ``kv_cache=`` on every hook (model_runner.py:750). The
        generator binds the cache by identity (``_binding``: ``id(kv)``), so the ``StackCache`` itself is returned;
        ``num_blocks`` sizes the page pool, block 0 is vLLM's null block (the tail of every page-table row)."""
        num_blocks, _, block_size, _ = (int(v) for v in shape)
        if block_size != BLOCK_SIZE:
            raise ValueError(f"vLLM block size {block_size} != the stack's {BLOCK_SIZE}")
        self.kv_cache = self.gen.allocate_kv_cache(
            self.max_batch, self.max_seq_len, num_blocks=num_blocks, null_block=0
        )
        self._alias = list(range(self.max_batch))  # a fresh cache: every generator slot is free
        return self.kv_cache

    def _kv(self, kv_cache):
        kv = kv_cache if kv_cache is not None else self.kv_cache
        if kv is None:
            raise RuntimeError("allocate_kv_cache must run before this hook")
        return kv

    # ================================================================================= warm-up
    def warmup_model_prefill(self, kv_cache=None, enable_trace: bool = True, can_sample_on_device: bool = False, **kw):
        """model_runner.py:4770-4799 phase 1 (``enable_trace=False``) -> ``warmup_serving(phase="prepare")``: every
        prefill bucket and the decode compiled eagerly, the idle state restored; phase 2 (``enable_trace=True``, before
        the decode hook's) -> ``phase="capture"``: the full-width decode trace and, behind it, the TP prefill traces
        (``StackBackend.capture_decode``). Both phases are idempotent in the generator, so the decode hook's calls are
        no-ops after these."""
        self.gen.warmup_serving(self._kv(kv_cache), phase="capture" if enable_trace else "prepare")

    def warmup_model_decode(
        self,
        kv_cache=None,
        enable_trace: bool = True,
        max_batch_size: int | None = None,
        num_blocks: int | None = None,
        can_sample_on_device: bool = False,
        **kw,
    ):
        """model_runner.py:4770-4799: the same two phases (prepared / captured already by ``warmup_model_prefill`` when
        the plugin traces prefill too; ``trace_mode=decode_only`` captures here). After the capture the one decode
        width is declared (model_runner.py:2301-2320: every decode is padded to ``max_num_seqs`` rows)."""
        if max_batch_size is not None and int(max_batch_size) != self.max_batch:
            raise ValueError(
                f"warmup_model_decode max_batch_size {max_batch_size} != the construction's {self.max_batch}"
            )
        self.gen.warmup_serving(self._kv(kv_cache), phase="capture" if enable_trace else "prepare")
        if enable_trace:
            self.tt_supported_decode_batch_sizes = (self.max_batch,)

    # ================================================================================= prefill
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        **kwargs,
    ):
        """model_runner.py:4009-4063 / MODEL_EXECUTION.md:37-61: row ``i`` of ``tokens [B, T]`` (history from position
        0) is the chunk ``tokens[i, start_pos[i]:prompt_lens[i]]`` of the request in plugin slot ``empty_slots[i]``
        (allocated by the plugin: model_runner.py:1299-1364; a first chunk takes the lowest free generator slot, a
        continuation keeps its slot: ``_admit``). The generator takes chunk-local tokens with their length and the
        absolute start (a continuation at the slot's cached length, 64 / 128-aligned), page-table rows trimmed to the
        live blocks. Returns ``(ids [B, 1], None)`` with ``sampling_params`` (device greedy: the first token
        sampled inside the prefill trace; every row must be greedy, which the plugin guarantees for
        ``max_device_top_k=1``) or ``logits [B, 1, V]`` fp32 for host sampling, which the plugin also selects when a
        row is an intermediate chunk (its head output is discarded)."""
        mm = [k for k in kwargs if k.startswith(("pixel_values", "image_", "video_", "audio_", "mm_"))]
        if mm:
            raise NotImplementedError(f"multimodal prompt on the text-only stack adapter ({mm[0]})")
        kv = self._kv(kv_cache)
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens[None]
        B = int(tokens.shape[0])
        if prompt_lens is None:
            raise ValueError("prefill_forward needs prompt_lens (the exclusive end of every row's chunk)")
        ends = [int(v) for v in prompt_lens]
        starts = [0] * B if start_pos is None else [int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()]
        slots = list(range(B)) if empty_slots is None else [int(s) for s in empty_slots]
        if len(ends) != B or len(starts) < B or len(slots) < B:
            raise ValueError(f"prefill rows {B} vs prompt_lens {len(ends)} start_pos {len(starts)} slots {len(slots)}")
        starts, slots = starts[:B], slots[:B]
        lens = [e - s for s, e in zip(starts, ends)]
        if any(n < 1 for n in lens) or any(e > int(tokens.shape[1]) for e in ends):
            raise ValueError(
                f"prefill chunks [start, end) {list(zip(starts, ends))} outside tokens [{B}, {tokens.shape[1]}]"
            )
        chunk = torch.zeros(B, max(lens), dtype=torch.int64)
        for i in range(B):
            chunk[i, : lens[i]] = tokens[i, starts[i] : ends[i]].to(torch.int64)
        if page_table is None:
            raise ValueError("prefill_forward needs the rows' page table")
        rows = trim_page_rows(page_table)
        if len(rows) < B:
            raise ValueError(f"prefill page_table has {len(rows)} rows for {B} prompts")
        rows = rows[:B]
        params = None
        if sampling_params is not None:
            params = [row_params(sampling_params, i) for i in range(B)]
            bad = [i for i, p in enumerate(params) if not p.is_greedy]
            if bad:
                raise ValueError(
                    f"prefill rows {bad} are sampled (temperature > 0 without top_k 1): the stack samples greedily on "
                    "device only; the plugin routes such rows to host sampling under max_device_top_k=1"
                )
        out = self.gen.prefill_forward(
            chunk,
            page_table=rows,
            kv_cache=kv,
            prompt_lens=lens,
            empty_slots=self._admit(kv, slots, starts),
            start_pos=starts,
            sampling_params=params,
        )
        if params is not None:
            return out.reshape(B, 1).to(torch.int32), None
        return out.reshape(B, 1, -1).float()

    # ================================================================================= decode
    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        slot_remap=None,
        *,
        reload_inputs: bool = True,
        reload_page_table: bool = False,
        reload_sampling_params: bool = False,
        reset_sampling_state: bool = False,
        **kwargs,
    ):
        """async_decode.py:1335-1440: one decode step of ``tokens [B_pad, 1]`` / ``start_pos [B_pad]`` (-1 on padding
        rows, ``B_pad`` = the declared width) where row ``i`` is plugin slot ``i`` once ``slot_remap`` (row ``i`` reads
        plugin slot ``remap[i]``, a full permutation: model_runner.py:1366-1421) is applied -- absorbed into the slot
        alias (``_permuted``): every row decodes in the generator slot its request lives in (``slots=``), nothing is
        copied, and the alias advances only once the generator accepted the step (the plugin commits its own map then:
        ``note_decode_state_slots_settled``). ``reload_inputs`` is always True for this adapter (async_decode.py:423-481: no ``supports_async_decode``), so tokens / positions /
        page rows are restaged from the host (the generator writes only what changed). Device route (``sampling_params``
        given): ``read_from_device=False`` returns the generator's ``DecodeOutput`` handle (DP=1 always sends it:
        model_runner.py:4106-4108), ``True`` the ids ``[B_pad, 1]``. Host route (no ``sampling_params``): the logits
        are read back regardless (the generator needs ``read_from_device=True`` without a sampler) and returned as
        ``(logits [B_pad, 1, V] fp32, None)``, which the plugin reads as ``tt_out[rows, -1, :]``
        (model_runner.py:4239-4245)."""
        if "reset_batch" in kwargs:
            raise TypeError("legacy reset_batch keyword: this adapter declares decode_input_update_contract = 1")
        if kwargs.get("spec_mode") is not None or kwargs.get("num_valid_drafts") is not None:
            raise NotImplementedError(
                "speculative decode is not declared by the stack adapter (supports_spec_decode False)"
            )
        if not reload_inputs:
            raise ValueError(
                "reload_inputs=False: the stack adapter declares no resident-input chain (supports_async_decode is "
                "False), every decode reloads its inputs"
            )
        kv = self._kv(kv_cache)
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        positions = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
        B = int(tokens.numel())
        if int(positions.numel()) != B:
            raise ValueError(f"tokens {B} rows vs start_pos {int(positions.numel())}")
        if B > self.max_batch:
            raise ValueError(f"{B} decode rows exceed the {self.max_batch} slots")
        rows = None
        if page_table is not None:
            rows = trim_page_rows(page_table)
            if int(rows.shape[0]) < B:
                raise ValueError(f"page_table {int(rows.shape[0])} rows < {B} decode rows")
            rows = rows[:B]
        alias = self._permuted(slot_remap)
        common = dict(
            page_table=rows,
            kv_cache=kv,
            slots=alias[:B],
            enable_trace=enable_trace,
            reload_inputs=True,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
        )
        if sampling_params is None:
            logits = self.gen.decode_forward(tokens, positions, sampling_params=None, read_from_device=True, **common)
            self._alias = alias
            return logits.reshape(B, 1, -1).float(), None
        params = [row_params(sampling_params, i) for i in range(B)]
        bad = [i for i, p in enumerate(params) if not p.is_greedy and int(positions[i]) >= 0]
        if bad:
            raise ValueError(
                f"decode rows {bad} are sampled (temperature > 0 without top_k 1): the stack samples greedily on device "
                "only; the plugin routes such rows to host sampling under max_device_top_k=1"
            )
        out = self.gen.decode_forward(
            tokens, positions, sampling_params=params, read_from_device=read_from_device, **common
        )
        self._alias = alias
        if isinstance(out, DecodeOutput):
            return out
        return self.gen.process_decode_output_host(out, is_tokens=True).to(torch.int32)

    # ================================================================================= slot aliasing
    def _permuted(self, slot_remap) -> list:
        """The alias after the plugin's gather ``slot_remap`` (row ``i`` reads plugin slot ``remap[i]``: a full
        permutation of the plugin's slots, model_runner.py:1366-1421; ``-1`` / ``i`` = unchanged; a ``{src: dst}``
        dict = those moves, the overwritten free slots backfilling the emptied rows), the current alias without one.
        Row ``i`` keeps the generator slot its request already decodes in, so no state moves."""
        n = self.max_batch
        if slot_remap is None:
            return self._alias
        if isinstance(slot_remap, dict):
            moves = {int(s): int(d) for s, d in slot_remap.items()}
            if any(not (0 <= s < n and 0 <= d < n) for s, d in moves.items()):
                raise ValueError(f"slot_remap moves {moves} outside [0, {n})")
            remap = list(range(n))
            for s, d in moves.items():
                remap[d] = s
            lost = [d for d in sorted(moves.values()) if d not in moves]  # free slots a move overwrote
            emptied = [s for s in sorted(moves) if s not in moves.values()]  # rows whose request moved away
            for i, d in zip(emptied, lost):
                remap[i] = d
        else:
            remap = [int(x) for x in torch.as_tensor(slot_remap).reshape(-1).tolist()]
            if len(remap) > n:
                raise ValueError(f"slot_remap names {len(remap)} rows for {n} slots")
            remap = [i if v < 0 else v for i, v in enumerate(remap)] + list(range(len(remap), n))
        if sorted(remap) != list(range(n)):
            raise ValueError(f"slot_remap {remap} is not a permutation of the {n} slots")
        return [self._alias[s] for s in remap]

    def _admit(self, kv, slots: list, starts: list) -> list:
        """The generator slots of a prefill batch's plugin slots. A new request (chunk start 0 in a plugin slot whose
        generator slot holds none) takes the lowest free generator slot: its alias entry is swapped with the free
        plugin slot that held that generator slot, invisible to the plugin, so the live generator slots stay as low as
        the plugin's rows (the generator's decode width covers slots 0..max live). A continuation keeps its slot."""
        n = self.max_batch
        if any(not 0 <= s < n for s in slots):
            raise ValueError(f"empty_slots {slots} outside [0, {n})")
        live = set(self.gen.live_slots(kv))
        alias, taken = list(self._alias), set(live)
        for s, start in zip(slots, starts):
            g = alias[s]
            if g in live or start > 0:
                taken.add(g)
                continue
            lowest = next(x for x in range(n) if x not in taken)
            if lowest != g:
                j = alias.index(lowest)
                alias[s], alias[j] = lowest, g
            taken.add(lowest)
        self._alias = alias
        return [alias[s] for s in slots]

    def read_decode_output(self, tt_out, async_read: bool = False, sample_rows=None):
        """async_decode.py:1474-1486 (the async read of a ``read_from_device=False`` submission): the generator
        enqueues the copy of the sampled rows behind the step and returns ``(PendingTokens, events)``; the plugin waits
        on the events, then ``process_decode_output_host`` yields the ids. A host output passes through. Selective
        readback (``sample_rows``) is not declared."""
        if sample_rows is not None:
            raise NotImplementedError("selective host readback is not declared by the stack adapter")
        if _is_host_output(tt_out):
            return (tt_out, []) if async_read else tt_out
        if not isinstance(tt_out, DecodeOutput):
            raise TypeError(f"read_decode_output expects a DecodeOutput, got {type(tt_out).__name__}")
        return self.gen.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out, is_tokens: bool = False, sample_rows=None):
        """async_decode.py:1651-1666: the ids ``[B_pad, 1]`` int32 of a device-sampled step (padding rows 0). The
        synchronous path hands ``decode_forward``'s raw ``DecodeOutput`` here, so it is read first
        (``Glm53Generator.process_decode_output_host`` takes ``PendingTokens`` or a tensor); a host output passes
        through. Compact host logits (``sample_rows``) are not declared."""
        if sample_rows is not None:
            raise NotImplementedError("compact host logits are not declared by the stack adapter")
        if _is_host_output(tt_out):
            return tt_out
        if isinstance(tt_out, DecodeOutput):
            tt_out = self.gen.read_decode_output(tt_out)
        if not isinstance(tt_out, (PendingTokens, torch.Tensor)):
            raise TypeError(
                f"process_decode_output_host expects a DecodeOutput / PendingTokens, got {type(tt_out).__name__}"
            )
        return self.gen.process_decode_output_host(tt_out, is_tokens=True).to(torch.int32)

    # ================================================================================= request lifecycle
    def release_request(self, slot: int) -> None:
        """model_runner.py:965-983 / 2880: the request in plugin ``slot`` (its CURRENT slot, after every gather)
        finished or was preempted -> ``Glm53Generator.release_slot`` on its generator slot: the slot stops decoding,
        its page coverage and (sharded pool) physical blocks are released, the backend's release hook runs once."""
        if self.kv_cache is None or not 0 <= int(slot) < self.max_batch:
            logger.warning(f"GLM53 stack vLLM: release_request({slot}) before allocation or outside the slot range")
            return
        self.gen.release_slot(self.kv_cache, self._alias[int(slot)])

    def note_state_slots_moved(self, moves) -> None:
        """model_runner.py:282 / MODEL_CAPABILITIES.md:73-75: the plugin's ``old -> new`` notification of the gather
        ``decode_forward`` already absorbed into the slot alias -- nothing to do (no state moved)."""
        return None

    def release_persistent_capture(self) -> None:
        """model_runner.py:525-540 (teardown, before the mesh closes): every captured trace is released
        (``Glm53Generator.release_traces``) and the decode width declaration withdrawn."""
        self.gen.release_traces()
        self.tt_supported_decode_batch_sizes = ()

    # ================================================================================= vLLM protocol shims
    # vLLM >= 0.30 admits the generate runner only for classes passing ``is_vllm_model`` (an ``__init__`` accepting
    # ``vllm_config``, ``embed_input_ids``, ``forward(input_ids, positions)``, ``compute_logits``); never called.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models do not expose embed_input_ids")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models run through prefill_forward / decode_forward")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models return logits from decode_forward")
