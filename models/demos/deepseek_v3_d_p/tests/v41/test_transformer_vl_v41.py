# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill of merged (image + text) prompts (bead 10.2): image span merge, VL routing bias, Engram mask.

Acceptance: merged-sequence blocks meet the text block acceptance, so the gate is ``test_transformer_v41._check``
unchanged (each layer's free-running streams, all rows and the scored text tail, vs the reference's worst noisy drift
minus DRIFT_MARGIN; bit-identical repeats; token agreement reported), on prompts whose first chunk holds image spans.
The reference is ``oracle.build_vl_reference``: the text reference plus each layer's ``bias_vl`` and the span
delimiters, run through ``Transformer.forward(images, token_types)``. Image features are teacher-forced (the same
aligner rows go to the reference and the device), so the gate measures the backbone on the merged sequence; the
encoder has its own bar (10.1) and the end-to-end image path is 10.3. The merged embedding itself is checked
bit-exact against the reference's (the merge copies rows).

Cases: small dims, two images (sharing roles; Engram layers 0 1 2 3), one chunk and two chunks; production shape
(S=2048, one 640x480 image, 206 span positions) with synthetic and real weights (real: the checkpoint's aligner rows
of the image). Precompute the CPU oracles first: ``scripts/precompute_vl_oracles.py`` in the bead's artifacts.
"""


import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41 import vl_prompts
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tests.v41.test_transformer_v41 import (
    MESH,
    NOISE,
    NOISE_SEEDS,
    PRODUCTION_CANDIDATE_BLOCKS,
    SCHEDULES,
    SCORED,
    _check,
    _stage,
)
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.engram import TtV41Engram, V41EngramHash, V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import (
    dequant_fp8_block,
    load_layer,
    load_layer_dense,
    resolve_checkpoint,
)

DELIMITERS = ("image_start", "image_end", "image_newline")


class _MergedPrompt:
    """A ``TtV41Transformer`` whose ``prefill`` carries one merged prompt's image inputs, so ``_check`` drives it."""

    def __init__(self, model: TtV41Transformer, prompt: orc.VLPrompt):
        self.model, self.mesh_device, self.config = model, model.mesh_device, model.config
        self.token_types = prompt.token_types
        replicate = ttnn.ReplicateTensorToMesh(model.mesh_device)
        self.features = [
            ttnn.from_torch(
                f[None, None],
                device=model.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate,
            )
            for f in prompt.features
        ]

    def prefill(self, tokens, logit_positions=1, on_block=None):
        return self.model.prefill(tokens, logit_positions, on_block, self.token_types, self.features)

    def merged_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """The first chunk's embedding after the image merge, [chunk, hidden] bf16 on host."""
        m, chunk = self.model, self.model.chunk
        ids = torch.zeros(chunk, dtype=torch.int64)
        ids[: min(chunk, tokens.numel())] = tokens[:chunk]
        types = torch.full((chunk,), -1, dtype=torch.int64)
        types[: min(chunk, tokens.numel())] = self.token_types[:chunk]
        rows = m._column(types != -1)
        h = m._merge_images(m.embedding(m._token_ids(ids)), types, self.features, rows)
        concat = ttnn.ConcatMesh2dToTensor(m.mesh_device, tuple(m.mesh_device.shape), dims=(2, 3))
        return ttnn.to_torch(h, mesh_composer=concat)[0, 0]


def _reference_unless_cached(spec, tokens: torch.Tensor, prompt: orc.VLPrompt):
    """None when every oracle ``_check`` reads is on disk (a real-weight reference build takes minutes), else the VL
    reference: the oracle functions would otherwise build the text reference, which ignores the images."""
    scored = min(SCORED, tokens.shape[1])
    paths = [orc.cache_path(spec, tokens), orc._noisy_path(spec, tokens, scored, None, "tail")]
    for seed in range(NOISE_SEEDS):
        paths += [orc._noisy_path(spec, tokens, scored, (*NOISE, seed), kind) for kind in ("tail", "drift")]
    return None if all(p.is_file() for p in paths) else orc.build_vl_reference(spec, prompt)


def _check_merge(merged: _MergedPrompt, tokens: torch.Tensor, result: dict, layer: int, name: str):
    """The merged embedding equals the reference's (its first block input, stream 0) bit for bit."""
    chunk = merged.model.chunk
    device = merged.merged_embedding(tokens[0])
    expected = result["blocks"][layer]["x_in"][:chunk, 0]
    assert torch.equal(device[: expected.shape[0]].to(expected.dtype), expected), f"{name}: merged embedding differs"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("schedule", ["sharing", "engram"])
@pytest.mark.parametrize("case", ["one_chunk", "two_chunks"])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_transformer_vl_small(mesh_device, device_params, case, schedule):
    layers = SCHEDULES[schedule]
    seq = vl_prompts.SMALL_SEQ
    chunk = {"one_chunk": seq, "two_chunks": seq // 2}[case]
    base = small_spec(layers, seq)
    tokens, prompt = vl_prompts.small_prompt(SmallV41Config.EMB_SIZE)
    spec = orc.vl_spec(base, prompt)
    with _stage(f"vl {schedule} {case} reference"):
        reference = orc.build_vl_reference(spec, prompt)
        orc.load_engram_rows(reference, spec, tokens)  # the rows of the masked hash (image tokens are DEAD)
    engram, engram_hash = {}, None
    if reference.engram_hash is not None:
        engram_hash = V41EngramHash(SmallV41Config, reference.engram_hash.token_map)
        for pos, layer in enumerate(layers):
            e = reference.layers[pos].engram
            if e is None:
                continue
            weights = {
                "wkv": dequant_fp8_block(e.wkv.weight.detach(), e.wkv.scale.detach()),
                "q_weight": e.q_weight.detach(),
                "k_weight": e.k_weight.detach(),
            }
            table = V41EngramTable(e.embed.weight.detach(), e.embed.scale.detach(), e.embed.oracle_rows)
            engram[layer] = TtV41Engram(mesh_device, SmallV41Config, layer, weights, table)

    def layer_weights(layer, include_moe):
        pos = layers.index(layer)
        weights = device_weights(reference, pos, include_moe)
        return weights | {"gate_bias_vl": reference.layers[pos].ffn.gate.bias_vl.detach()}

    # the text weights are the text spec's: its device MoE tensors are reused
    with _stage(f"vl {schedule} {case} build"):
        model = TtV41Transformer(
            mesh_device,
            SmallV41Config,
            list(layers),
            layer_weights,
            reference.embed.weight.detach(),
            reference.norm.weight.detach(),
            reference.head.weight.detach(),
            max_seq_len=seq,
            chunk=chunk,
            engram=engram,
            engram_hash=engram_hash,
            image_embeds={k: getattr(reference, k).detach() for k in DELIMITERS},
            weight_cache_path=weight_cache_dir(base, mesh_device.shape),
        )
    merged = _MergedPrompt(model, prompt)
    with _stage(f"vl {schedule} {case} merge exactness"):
        _check_merge(merged, tokens, orc.oracle(spec, tokens, reference), layers[0], f"vl {schedule} {case}")
    _check(merged, spec, tokens, reference, f"vl {schedule} {case}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("weights", ["synthetic", "real"])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_transformer_vl_production(mesh_device, device_params, weights):
    """Real dims, layers 0 2 3 20 21 24, S=2048 in one chunk, one 640x480 image. Precompute the reference first."""
    layers = SCHEDULES["sharing"]
    ckpt = resolve_checkpoint() if weights == "real" else None
    if weights == "real" and ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    seq = vl_prompts.PRODUCTION_SEQ
    base = orc.real_spec(
        layers, seq, candidate_topk_blocks=PRODUCTION_CANDIDATE_BLOCKS, checkpoint=ckpt.root if ckpt else None
    )
    cfg = type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": PRODUCTION_CANDIDATE_BLOCKS})
    with _stage(f"vl production {weights} prompt (vision oracle cached)"):
        tokens, prompt, _ = vl_prompts.production_prompt(ckpt.root if ckpt else None)
    spec = orc.vl_spec(base, prompt)
    if ckpt is None:
        reference = orc.build_vl_reference(spec, prompt)

        def layer_weights(layer, include_moe):
            pos = layers.index(layer)
            weights = device_weights(reference, pos, include_moe)
            return weights | {"gate_bias_vl": reference.layers[pos].ffn.gate.bias_vl.detach()}

        embed, norm, head = (
            reference.embed.weight.detach(),
            reference.norm.weight.detach(),
            reference.head.weight.detach(),
        )
        image_embeds = {k: getattr(reference, k).detach() for k in DELIMITERS}
    else:
        reference = _reference_unless_cached(spec, tokens, prompt)
        layer_weights = lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer)
        top = ckpt.read(["embed.weight", "norm.weight", "head.weight", *DELIMITERS])
        embed, norm, head = top["embed.weight"], top["norm.weight"], top["head.weight"]
        image_embeds = {k: top[k] for k in DELIMITERS}
    with _stage(f"vl production {weights} build (MoE weights cached after the first build)"):
        model = TtV41Transformer(
            mesh_device,
            cfg,
            list(layers),
            layer_weights,
            embed,
            norm,
            head,
            max_seq_len=seq,
            chunk=seq,
            image_embeds=image_embeds,
            weight_cache_path=weight_cache_dir(base, mesh_device.shape),
        )
    merged = _MergedPrompt(model, prompt)
    with _stage(f"vl production {weights} merge exactness"):
        _check_merge(merged, tokens, orc.oracle(spec, tokens, reference), layers[0], f"vl production {weights}")
    _check(merged, spec, tokens, reference, f"vl production {weights}")
