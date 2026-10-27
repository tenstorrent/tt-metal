# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side preparation of the DSpark drafter's checkpoint weights, the ``mtp.<stage>`` units (speculative
decode).

The checkpoint's drafter is three complete transformer stages under ``mtp.0`` to ``mtp.2``
(``reference/deepseek_v41/
model.py`` ``DSparkBlock``): each a ratio-0 (sliding-window) attention of the backbone's shape, a 128-expert
top-3
MoE of the backbone's intermediate size (``dspark_n_routed_experts`` / ``dspark_num_experts_per_tok`` in
``config.json``; ``ModelArgs.get_moe_config``) with one shared expert, and the mHC tables; stage 0 also holds
the
main projection ``main_proj`` ``[hidden, n_taps * hidden]`` and ``main_norm`` (``DSparkBlock.__init__``, stage
0),
stage 2 the final ``norm``, the Markov head's ``embed`` / ``head`` ``[vocab, markov_rank]`` and the confidence
head
``[1, hidden + markov_rank]`` (stage ``n_mtp_layers - 1``). The token embedding and the output head are the
backbone's (tied: ``Transformer.__init__`` assigns ``self.embed`` / ``self.head`` to every stage; the oracle
pops
``mtp.*.embed.weight`` / ``head.weight`` on load).

Formats are the backbone's (``weights.py``): dense projections ``F8_E4M3`` with ``F8_E8M0`` 32x32 block
scales,
routed experts MXFP4, norms and the gate bf16, biases / sink / hc tables fp32, the Markov tables and the
confidence
projection bf16. Every dequantization here is the backbone's function (``dequant_fp8_block``,
``dequant_mxfp4``,
``_prepare``), so the returned tensors are exact in bf16 as the backbone's are.

Device placement (the converters are the backbone's, cited per tensor):

* routed experts: expert-parallel over the 32 chips, ``experts_per_chip = 128 / 32 = 4`` a stage, in the mesh
order
  of ``ExpertMapping.get_global_expert_idx`` with ``dispatch_group_size`` = mesh rows (8) and
  ``num_dispatch_groups``
  = mesh columns (4), stacked and sharded by ``TtRoutedExpert._convert_and_cache_expert_weights`` (``[rows,
  cols, in,
  out]`` per local expert, ``ShardTensor2dMesh(dims=(0, 1))``) at the routed-expert dtype (bfp8 the default,
  ``TT_V41_EXPERT_DTYPE``), under the cache prefix ``layer_<40 + stage>.routed_expert`` (``TtMoe`` names its
  files
  by ``layer_idx``; the stages take the config's layer ids 40 to 42,
  ``DeepSeekV41FlashConfig.COMPRESS_RATIOS``
  covers them with ratio 0). Checkpoint expert order (``expert_placement.expert_order`` has no profile for
  these
  layers and returns None): no relabelling.
* the attention projections ``wq_a`` / ``wkv`` / ``wo_b`` row-parallel over TP
(``ShardTensor2dMesh(dims=(None,
  2))`` of the transposed ``[in, out]``), ``wq_b`` column-parallel (dims ``(None, 3)``), ``wo_a`` grouped
  (``[groups, in/groups, rank]`` sharded on the group axis): ``TtV41Attention.__init__``'s ``linear_weight``
  and the
  ``wo_a`` upload, bfp8; the norms and the sink replicated bf16 rows. The shapes are the backbone's, so
  ``TtV41Attention(mesh_device, config, 40 + stage, weights["attn"])`` places them (its forward is the
  target's
  causal one; the drafter's non-causal block attention is ``dspark_drafter.py``).
* ``main_proj`` row-parallel over TP in tap-major-to-chip-major order and ``main_norm`` replicated: already
  ``TtV41DSpark.__init__`` (``dspark.py``); ``seeding_weights`` returns its dict.
* the shared expert and the gate: ``TtMoe``'s (``TtSharedExpert._convert_and_cache_weights``,
  ``TtMoEGatePrefill._convert_and_cache_gate_weights``), the gate bias recentred as ``TtV41Moe.__init__``
  does.
* the Markov tables: ``head`` column-parallel over TP as the tied output head (``TtLMHead``, ``vocab / tp``
columns a
  chip, bf16: 16.5 MB a chip), ``embed`` replicated bf16 (``[vocab, 256]``, 66 MB a chip: one
  ``ttnn.embedding``
  of the previous draft token); the confidence projection replicated.
* the mHC tables as ``TtV41Block``'s (``mhc.py``).

The device-weight cache of the drafter is its own directory (``dspark_cache_dir``): keyed by the checkpoint,
the
routed-expert dtype, the mesh and a digest of the conversion code it uses, and separate from the backbone's
``weight_cache_dir`` so that adding the drafter changes nothing in the backbone's key (a key change would
reconvert its 40 layers, 1 to 2 minutes a layer).
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tt.v41 import weights as checkpoint_weights
from models.demos.deepseek_v3_d_p.tt.v41.weights import (
    _EXPERT_PROJ,
    BF16,
    EXPERT_READ_BATCH,
    F32,
    FP8,
    V41Checkpoint,
    _Param,
    _prepare,
    _stored_names,
    dequant_mxfp4,
)

DSPARK_UNIT = "mtp"
DSPARK_ROUTED_EXPERTS = 128  # config.json dspark_n_routed_experts
DSPARK_EXPERTS_PER_TOKEN = 3  # config.json dspark_num_experts_per_tok
DSPARK_MARKOV_RANK = 256  # config.json dspark_markov_rank
DSPARK_BLOCK_SIZE = 5  # config.json dspark_block_size (draft block rows: x[p] + 4 noise tokens)
DSPARK_NOISE_TOKEN_ID = 128799  # config.json dspark_noise_token_id


def dspark_layer_id(stage: int, config=DeepSeekV41FlashConfig) -> int:
    """The config's layer id of DSpark stage ``stage`` (40 to 42: ``COMPRESS_RATIOS`` holds their ratio 0; the
    ``TtMoe`` cache prefix ``layer_<id>`` and ``TtV41Attention``'s ``layer`` take it)."""
    _check_stage(stage, config)
    return config.NUM_LAYERS + stage


def _check_stage(stage: int, config) -> None:
    if not 0 <= stage < config.NUM_DSPARK_LAYERS:
        raise ValueError(f"stage {stage} is not a DSpark stage (0..{config.NUM_DSPARK_LAYERS - 1})")


def _stage_params(config, stage: int) -> list[_Param]:
    """Every non-routed-expert tensor of DSpark stage ``stage``, by role (the backbone's ``_dense_params``
    minus the
    compressor / indexer, which a ratio-0 stage has none of, plus the stage-specific heads)."""
    dim, q_lora, heads, head_dim = config.EMB_SIZE, config.Q_LORA_RANK, config.NUM_ATTENTION_HEADS, config.HEAD_DIM
    o_rank, o_groups, inter = config.O_LORA_RANK * config.O_GROUPS, config.O_GROUPS, config.MOE_INTERMEDIATE_SIZE
    shared_inter = inter * config.NUM_SHARED_EXPERTS
    experts, hc = DSPARK_ROUTED_EXPERTS, config.HC_MULT
    mix_hc = (2 + hc) * hc
    P = _Param
    params = [
        P(("attn", "wq_a"), "attn.wq_a", FP8, (q_lora, dim)),
        P(("attn", "q_norm"), "attn.q_norm.weight", BF16, (q_lora,)),
        P(("attn", "wq_b"), "attn.wq_b", FP8, (heads * head_dim, q_lora)),
        P(("attn", "wkv"), "attn.wkv", FP8, (head_dim, dim)),
        P(("attn", "kv_norm"), "attn.kv_norm.weight", BF16, (head_dim,)),
        P(("attn", "wo_a"), "attn.wo_a", FP8, (o_rank, heads * head_dim // o_groups)),
        P(("attn", "wo_b"), "attn.wo_b", FP8, (dim, o_rank)),
        P(("attn", "attn_sink"), "attn.attn_sink", F32, (heads,)),
        P(("attn_norm",), "attn_norm.weight", BF16, (dim,)),
        P(("ffn_norm",), "ffn_norm.weight", BF16, (dim,)),
        P(("gate_weights", "weight"), "ffn.gate.weight", BF16, (experts, dim)),
        P(("gate_weights", "e_score_correction_bias"), "ffn.gate.bias", F32, (experts,)),
        P(("gate_bias_vl",), "ffn.gate.bias_vl", F32, (experts,)),
        P(("shared_expert_weights", "gate_proj"), "ffn.shared_experts.w1", FP8, (shared_inter, dim)),
        P(("shared_expert_weights", "up_proj"), "ffn.shared_experts.w3", FP8, (shared_inter, dim)),
        P(("shared_expert_weights", "down_proj"), "ffn.shared_experts.w2", FP8, (dim, shared_inter)),
    ]
    for site in ("attn", "ffn"):
        params += [
            P((f"hc_{site}", "fn"), f"hc_{site}_fn", F32, (mix_hc, hc * dim)),
            P((f"hc_{site}", "base"), f"hc_{site}_base", F32, (mix_hc,)),
            P((f"hc_{site}", "scale"), f"hc_{site}_scale", F32, (3,)),
        ]
    if stage == 0:
        n_taps = len(config.DSPARK_TARGET_LAYER_IDS)
        params += [
            P(("main_proj",), "main_proj", FP8, (dim, n_taps * dim)),
            P(("main_norm",), "main_norm.weight", BF16, (dim,)),
        ]
    if stage == config.NUM_DSPARK_LAYERS - 1:
        vocab, rank = config.VOCAB_SIZE, DSPARK_MARKOV_RANK
        params += [
            P(("norm",), "norm.weight", BF16, (dim,)),
            P(("markov", "embed"), "markov_head.embed.weight", BF16, (vocab, rank)),
            P(("markov", "head"), "markov_head.head.weight", BF16, (vocab, rank)),
            P(("confidence",), "confidence_head.proj.weight", BF16, (1, dim + rank)),
        ]
    return params


def stage_prefix(stage: int) -> str:
    return f"{DSPARK_UNIT}.{stage}."


def stage_tensor_names(config, stage: int) -> list[str]:
    """Every checkpoint tensor name of ``stage`` (dense parameters and routed experts), the schema this module
    reads; ``V41Checkpoint.weight_map`` must hold each (``test_dspark_weights.py`` checks it against the
    index)."""
    _check_stage(stage, config)
    prefix = stage_prefix(stage)
    names = [n for p in _stage_params(config, stage) for n in _stored_names(p, prefix)]
    for e in range(DSPARK_ROUTED_EXPERTS):
        for _, w, _ in _EXPERT_PROJ:
            names += [f"{prefix}ffn.experts.{e}.{w}.weight", f"{prefix}ffn.experts.{e}.{w}.scale"]
    return names


def load_stage_dense(ckpt: V41Checkpoint, stage: int, config=DeepSeekV41FlashConfig) -> dict:
    """Everything of DSpark ``stage`` except the routed experts, keyed as ``weights.load_layer_dense`` keys
    the
    backbone's (``attn``, ``attn_norm``, ``ffn_norm``, ``hc_attn`` / ``hc_ffn`` tuples, ``gate_weights``,
    ``gate_bias_vl``, ``shared_expert_weights``), plus ``main_proj`` / ``main_norm`` (stage 0) and ``norm``,
    ``markov`` {embed, head}, ``confidence`` (the last stage). bf16 unless fp32 as in the backbone."""
    _check_stage(stage, config)
    prefix = stage_prefix(stage)
    params = _stage_params(config, stage)
    raw = ckpt.read([n for p in params for n in _stored_names(p, prefix)])
    out: dict = {}
    for p in params:
        node = out
        for k in p.key[:-1]:
            node = node.setdefault(k, {})
        node[p.key[-1]] = _prepare(p, raw, prefix)
    for site in ("hc_attn", "hc_ffn"):
        out[site] = (out[site]["fn"], out[site]["base"], out[site]["scale"])
    return out


def load_stage_experts(
    ckpt: V41Checkpoint, stage: int, config=DeepSeekV41FlashConfig, experts: range | None = None
) -> list[dict]:
    """``[{gate_proj: w1, up_proj: w3, down_proj: w2}]`` bf16 per routed expert of ``stage`` in checkpoint
    order
    (``experts``: a sub-range, for tests; default all 128). The backbone's ``load_routed_experts`` with the
    stage's
    prefix and expert count: the same shapes (``[inter, dim]`` / ``[dim, inter]``), the same
    ``dequant_mxfp4``."""
    _check_stage(stage, config)
    dim, inter = config.EMB_SIZE, config.MOE_INTERMEDIATE_SIZE
    prefix = stage_prefix(stage)
    experts = range(DSPARK_ROUTED_EXPERTS) if experts is None else experts
    out = []
    ids_all = list(experts)
    for start in range(0, len(ids_all), EXPERT_READ_BATCH):
        ids = ids_all[start : start + EXPERT_READ_BATCH]
        stems = {(e, key): f"{prefix}ffn.experts.{e}.{w}" for e in ids for key, w, _ in _EXPERT_PROJ}
        raw = ckpt.read([f"{s}.{part}" for s in stems.values() for part in ("weight", "scale")])
        for e in ids:
            entry = {}
            for key, _, down in _EXPERT_PROJ:
                stem = stems[(e, key)]
                t = dequant_mxfp4(raw.pop(f"{stem}.weight"), raw.pop(f"{stem}.scale"), stem)
                expected = (dim, inter) if down else (inter, dim)
                if tuple(t.shape) != expected:
                    raise ValueError(f"{stem}: shape {tuple(t.shape)}, expected {expected}")
                entry[key] = t
            out.append(entry)
    return out


def load_stage(ckpt: V41Checkpoint, stage: int, config=DeepSeekV41FlashConfig) -> dict:
    """``load_stage_dense`` plus ``routed_expert_weights``: a stage's full weights dict (the
    ``TtV41Block``-shaped
    entries plus the drafter's heads)."""
    weights = load_stage_dense(ckpt, stage, config)
    weights["routed_expert_weights"] = load_stage_experts(ckpt, stage, config)
    return weights


def seeding_weights(ckpt: V41Checkpoint, config=DeepSeekV41FlashConfig) -> dict:
    """The ``TtV41DSpark`` ``weights`` dict from the checkpoint (``main_proj`` ``[hidden, n_taps * hidden]``,
    ``main_norm``, ``layers`` of ``wkv`` / ``kv_norm`` per stage): the seven tensors the prefill seeding
    reads,
    without the stages' other 2,394 (``tests/v41/test_dspark.py`` builds the same dict from the reference
    module)."""
    P = _Param
    dim, head_dim = config.EMB_SIZE, config.HEAD_DIM
    n_taps = len(config.DSPARK_TARGET_LAYER_IDS)
    main = [
        P(("main_proj",), "main_proj", FP8, (dim, n_taps * dim)),
        P(("main_norm",), "main_norm.weight", BF16, (dim,)),
    ]
    per_stage = [
        P(("wkv",), "attn.wkv", FP8, (head_dim, dim)),
        P(("kv_norm",), "attn.kv_norm.weight", BF16, (head_dim,)),
    ]
    names = [n for p in main for n in _stored_names(p, stage_prefix(0))]
    for k in range(config.NUM_DSPARK_LAYERS):
        names += [n for p in per_stage for n in _stored_names(p, stage_prefix(k))]
    raw = ckpt.read(names)
    out = {p.key[0]: _prepare(p, raw, stage_prefix(0)) for p in main}
    out["layers"] = [
        {p.key[0]: _prepare(p, raw, stage_prefix(k)) for p in per_stage} for k in range(config.NUM_DSPARK_LAYERS)
    ]
    return out


# --- the drafter's device-weight cache directory
# -------------------------------------------------------------

# the conversion code the drafter's stored device tensors depend on (as tests/v41/weight_cache.py keys the
# backbone's)
_CONVERSION_CODE = (
    checkpoint_weights.dequant_fp8_block,
    checkpoint_weights.dequant_mxfp4,
    checkpoint_weights._prepare,
    _stage_params,
    load_stage_dense,
    load_stage_experts,
    seeding_weights,
)


def dspark_conversion_digest(extra=()) -> str:
    """Digest of the conversion functions above plus ``extra`` (the device converters the caller uses:
    ``TtRoutedExpert._convert_and_cache_expert_weights`` and the like, passed in so this module needs no
    ttnn)."""
    code = "\n".join(inspect.getsource(f) for f in (*_CONVERSION_CODE, *extra)).encode()
    return hashlib.sha256(code).hexdigest()[:20]


def dspark_cache_dir(root: Path, mesh_shape, routed_expert_dtype_name: str, extra=()) -> Path:
    """The drafter's device-weight cache directory under ``root`` (``TT_V41_WEIGHT_CACHE``):
    ``dspark-<key>-mesh<sp>x
    <tp>``, the key a digest of the checkpoint identity, the routed-expert dtype name (``BFLOAT8_B`` /
    ``BFLOAT4_B``) and the conversion digest. Never the backbone's ``real-<key>-mesh..`` directory."""
    sp, tp = tuple(mesh_shape)
    source = f"hf:{checkpoint_weights.CHECKPOINT_REPO}@{checkpoint_weights.CHECKPOINT_REVISION}"
    key = hashlib.sha256(
        "|".join([source, routed_expert_dtype_name, dspark_conversion_digest(extra), f"{sp}x{tp}"]).encode()
    ).hexdigest()[:20]
    return Path(root) / f"dspark-{key}-mesh{sp}x{tp}"


def split_tp_rows(w: torch.Tensor, tp: int) -> list[torch.Tensor]:
    """``TtV41Attention.linear_weight(w, 2)``'s per-chip shards of a checkpoint ``[out, in]`` projection: the
    transposed ``[in, out]`` cut into ``tp`` row blocks (row-parallel over the input)."""
    t = w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()
    assert t.shape[0] % tp == 0, (tuple(t.shape), tp)
    return list(t.chunk(tp, dim=0))


def split_tp_cols(w: torch.Tensor, tp: int) -> list[torch.Tensor]:
    """``TtV41Attention.linear_weight(w, 3)``'s shards: the transposed ``[in, out]`` cut into ``tp`` column
    blocks
    (column-parallel over the output; ``wq_b``: the heads)."""
    t = w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()
    assert t.shape[1] % tp == 0, (tuple(t.shape), tp)
    return list(t.chunk(tp, dim=1))


def main_proj_shards(w: torch.Tensor, n_taps: int, tp: int) -> list[torch.Tensor]:
    """``TtV41DSpark.__init__``'s ``main_proj`` per-chip shards: the transposed ``[n_taps * hidden, hidden]``
    reordered from (tap, chip, j) rows to (chip, tap, j), so TP shard ``t`` multiplies the local concatenation
    ``[tap_0 slice t | tap_1 slice t | ...]`` of the SP-gathered taps (``dspark.py``)."""
    hidden = w.shape[0]
    t = w.detach().to(torch.bfloat16).T
    assert tuple(t.shape) == (n_taps * hidden, hidden)
    t = t.reshape(n_taps, tp, hidden // tp, hidden).transpose(0, 1).reshape(n_taps * hidden, hidden).contiguous()
    return list(t.chunk(tp, dim=0))


def expert_chip(expert: int, mesh_rows: int, mesh_cols: int, experts_per_chip: int) -> tuple[int, int, int]:
    """(mesh row, mesh column, local index) holding checkpoint expert ``expert`` under the backbone's
    placement
    (``ExpertMapping.get_global_expert_idx``, column-major groups: chip ``row`` of group ``col`` holds experts
    ``(col * mesh_rows + row) * experts_per_chip + local``)."""
    chip, local = divmod(expert, experts_per_chip)
    col, row = divmod(chip, mesh_rows)
    assert col < mesh_cols, (expert, mesh_rows, mesh_cols, experts_per_chip)
    return row, col, local
