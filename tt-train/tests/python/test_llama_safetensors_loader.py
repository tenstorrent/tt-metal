# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Llama safetensors loader."""

from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

import ttnn
import ttml
from bf16_ulp import assert_within_bf16_ulp
from ttml.common.utils import build_causal_mask, unpermute_proj_rows
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors, safetensors_loader
from ttml.models.llama.safetensors_loader import (
    COL_DIM,
    ROW_DIM,
    _TIED_NAMES,
    _Checkpoint,
    _Rule,
    _TpAxis,
    _assemble,
    _assembled_shape,
    _canonical,
    _check_coverage,
    _check_sources,
    _load_plan,
    _pad_to,
    _rules,
    _sharded_dim,
    _tp_axis,
)

HEAD_DIM = 8  # even: RoPE splits each head into two halves
HIDDEN = 16

# (num_heads, num_kv_heads, tp_size) -- GQA, then MHA (one group per head) and MQA (one group).
QKV_CASES = [
    (8, 4, 1),
    (8, 4, 2),
    (8, 4, 4),
    (16, 8, 8),
    (8, 8, 2),
    (8, 1, 1),
]


def fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size):
    """What ``_rules`` asks the driver to do for ``qkv_linear``, without the device."""
    blocks = [unpermute_proj_rows(q, num_heads), unpermute_proj_rows(k, num_kv_heads), v]
    return _assemble(blocks, ROW_DIM, tp_size)


def fuse_gate_up(gate, up, tp_size):
    return _assemble([gate, up], ROW_DIM, tp_size)


def hf_qkv(num_heads: int, num_kv_heads: int, seed: int):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((num_heads * HEAD_DIM, HIDDEN)).astype(np.float32),
        rng.standard_normal((num_kv_heads * HEAD_DIM, HIDDEN)).astype(np.float32),
        rng.standard_normal((num_kv_heads * HEAD_DIM, HIDDEN)).astype(np.float32),
    )


def unfuse(fused: np.ndarray, block_rows: list[int], tp_size: int) -> list[np.ndarray]:
    """Inverse of the interleave."""
    per_rank = [rows // tp_size for rows in block_rows]
    shard_rows = sum(per_rank)
    assert fused.shape[0] == shard_rows * tp_size, "fused row count does not match the blocks"

    recovered: list[list[np.ndarray]] = [[] for _ in block_rows]
    for rank in range(tp_size):
        shard = fused[rank * shard_rows : (rank + 1) * shard_rows]
        offset = 0
        for i, rows in enumerate(per_rank):
            recovered[i].append(shard[offset : offset + rows])
            offset += rows
    return [np.concatenate(parts, axis=0) for parts in recovered]


class TestFuseQkv:
    @pytest.mark.parametrize("num_heads,num_kv_heads,tp_size", QKV_CASES)
    def test_every_shard_holds_matched_q_k_v(self, num_heads, num_kv_heads, tp_size):
        q, k, v = hf_qkv(num_heads, num_kv_heads, seed=1)
        fused = fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size)

        block_rows = [num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM, num_kv_heads * HEAD_DIM]
        got_q, got_k, got_v = unfuse(fused, block_rows, tp_size)

        assert np.array_equal(got_q, unpermute_proj_rows(q, num_heads)), "Q rows scrambled"
        assert np.array_equal(got_k, unpermute_proj_rows(k, num_kv_heads)), "K rows scrambled"
        assert np.array_equal(got_v, v), "V rows scrambled"

    @pytest.mark.parametrize("num_heads,num_kv_heads,tp_size", QKV_CASES)
    def test_each_rank_holds_the_groups_its_heads_need(self, num_heads, num_kv_heads, tp_size):
        """GQA runs locally over ``num_groups // tp`` groups, so co-locating a head with its
        group is what makes local attention equal the global one."""
        q, k, v = hf_qkv(num_heads, num_kv_heads, seed=2)
        fused = fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size)

        heads_per_group = num_heads // num_kv_heads
        local_heads, local_groups = num_heads // tp_size, num_kv_heads // tp_size
        shard_rows = (local_heads + 2 * local_groups) * HEAD_DIM
        unpermuted_k = unpermute_proj_rows(k, num_kv_heads)
        assert fused.shape[0] == shard_rows * tp_size, "fused height disagrees with the local block sizes"

        for rank in range(tp_size):
            shard = fused[rank * shard_rows : (rank + 1) * shard_rows]
            local_k = shard[local_heads * HEAD_DIM :][: local_groups * HEAD_DIM]
            for local_head in range(local_heads):
                global_head = rank * local_heads + local_head
                local_group = local_head // heads_per_group
                global_group = global_head // heads_per_group
                assert np.array_equal(
                    local_k[local_group * HEAD_DIM : (local_group + 1) * HEAD_DIM],
                    unpermuted_k[global_group * HEAD_DIM : (global_group + 1) * HEAD_DIM],
                ), f"rank {rank} head {local_head} reads the wrong K group"


class TestFuseGateUp:
    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_every_shard_holds_matched_gate_up(self, tp_size):
        intermediate = 256
        rng = np.random.default_rng(5)
        gate = rng.standard_normal((intermediate, HIDDEN)).astype(np.float32)
        up = rng.standard_normal((intermediate, HIDDEN)).astype(np.float32)

        fused = fuse_gate_up(gate, up, tp_size)
        assert fused.shape == (2 * intermediate, HIDDEN)

        got_gate, got_up = unfuse(fused, [intermediate, intermediate], tp_size)
        assert np.array_equal(got_gate, gate), "gate rows scrambled"
        assert np.array_equal(got_up, up), "up rows scrambled"


class TestAssemble:
    """Interleaving follows from the destination's placement, not from the caller."""

    @pytest.mark.parametrize("shard_dim", [None, 3], ids=["replicated", "cols-sharded"])
    def test_only_row_sharding_interleaves(self, shard_dim):
        rng = np.random.default_rng(7)
        a, b = rng.standard_normal((8, 4)), rng.standard_normal((8, 4))
        plain = np.concatenate([a, b], axis=0)
        assert np.array_equal(_assemble([a, b], shard_dim, 4), plain)

    @pytest.mark.parametrize("shard_dim", [None, ROW_DIM, COL_DIM], ids=["replicated", "rows", "cols"])
    def test_the_shape_check_predicts_the_assembly(self, shard_dim):
        """``_assembled_shape`` must predict the shape ``_assemble`` actually builds."""
        blocks = [np.zeros((8, HIDDEN), np.float32), np.zeros((4, HIDDEN), np.float32)]
        shapes = [block.shape for block in blocks]
        assert _assembled_shape(shapes, shard_dim, 4, "w") == _assemble(blocks, shard_dim, 4).shape

    def test_rejects_rows_not_divisible_over_the_mesh(self, expect_error):
        with expect_error(RuntimeError, "not divisible over 4 devices"):
            _assembled_shape([(6, HIDDEN), (6, HIDDEN)], ROW_DIM, 4, "w_gate_up")

    def test_replicated_blocks_need_not_divide_the_mesh(self):
        """Only a row-shard splits, so an indivisible replicated weight is fine."""
        assert _assembled_shape([(3, HIDDEN), (3, HIDDEN)], None, 4, "w_gate_up") == (6, HIDDEN)

    def test_rejects_mismatched_widths(self, expect_error):
        with expect_error(RuntimeError, "columns, expected"):
            _assembled_shape([(4, HIDDEN), (4, HIDDEN + 1)], ROW_DIM, 2, "qkv_linear")


class TestPlacementWithoutTensorParallelism:
    """Single-device loads (GRPO, inference) take this path, with nothing sharded."""

    @pytest.mark.parametrize("mesh", [None, SimpleNamespace(has_axis=lambda _: False)], ids=["no mesh", "no tp axis"])
    def test_has_no_tp_axis(self, monkeypatch, mesh):
        monkeypatch.setattr(ttml, "maybe_mesh", lambda: mesh)
        assert _tp_axis() is None

    def test_rejects_a_shard_over_another_axis(self, monkeypatch, expect_error):
        """Assigning the whole tensor to a dp-sharded parameter would silently undo the sharding."""
        sharded = SimpleNamespace(placements=[ttnn.PlacementShard(ROW_DIM)])
        monkeypatch.setattr(ttml.Sharding, "from_tensor", lambda _: sharded)
        with expect_error(RuntimeError, "Llama/fc/weight: sharded over mesh axis 0"):
            _sharded_dim(object(), None, "Llama/fc/weight")

    def test_rejects_a_tp_shard_on_neither_rows_nor_columns(self, monkeypatch, expect_error):
        sharded = SimpleNamespace(placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(1)])
        monkeypatch.setattr(ttml.Sharding, "from_tensor", lambda _: sharded)
        with expect_error(RuntimeError, "Llama/fc/weight: sharded on dim 1 over 'tp'"):
            _sharded_dim(object(), _TpAxis(index=1, size=2), "Llama/fc/weight")


class HeadersOnly:
    """A fake checkpoint with shapes but no data, so any attempt to read a tensor fails."""

    def __init__(self, shapes: dict[str, tuple[int, int]]) -> None:
        self._shapes = shapes
        self.names = frozenset(shapes)

    def shape(self, name: str) -> tuple[int, int]:
        return self._shapes[name]


class TestLoadPlan:
    """Runs without a device: placements are stubbed, and shapes come from ``HeadersOnly``."""

    @staticmethod
    def param(rows: int, cols: int):
        return SimpleNamespace(shape=lambda: [1, 1, rows, cols])

    @pytest.fixture
    def placements(self, monkeypatch):
        """The placements every parameter reports; a test may change them in place."""
        current = [ttnn.PlacementReplicate()]
        monkeypatch.setattr(ttml.Sharding, "from_tensor", lambda _: SimpleNamespace(placements=current))
        return current

    def test_without_a_source_shape_requires_the_parameter_shape(self, placements, expect_error):
        with expect_error(RuntimeError, "does not match the config-implied shape"):
            _load_plan(_Rule("w", ("a",)), self.param(3, 4), None, 1, HeadersOnly({"a": (4, 3)}))

    def test_grows_the_config_vocab_to_the_tile_padded_parameter(self, placements):
        rule = _Rule("w", ("a",), source_shape=(120, 3))
        assert _load_plan(rule, self.param(128, 3), None, 1, HeadersOnly({"a": (120, 3)})) == (None, (128, 3))

    def test_rejects_a_vocab_that_disagrees_with_the_config(self, placements, expect_error):
        """A slightly short vocab still fits the tile-padded parameter, so only the config check catches it."""
        rule = _Rule("w", ("a",), source_shape=(128, 3))
        with expect_error(RuntimeError, "does not match the config-implied shape"):
            _load_plan(rule, self.param(128, 3), None, 1, HeadersOnly({"a": (120, 3)}))

    def test_a_row_shard_expects_the_global_shape(self, placements):
        placements[:] = [ttnn.PlacementReplicate(), ttnn.PlacementShard(ROW_DIM)]
        tp = _TpAxis(index=1, size=2)
        rule = _Rule("w_gate_up", ("gate", "up"), block_rows=(64, 64))
        headers = HeadersOnly({"gate": (64, HIDDEN), "up": (64, HIDDEN)})
        assert _load_plan(rule, self.param(64, HIDDEN), tp, 2, headers) == (ROW_DIM, (128, HIDDEN))

    @pytest.mark.parametrize(
        "suffix,rows",
        [("attention/qkv_linear/weight", (96, 96, 64)), ("mlp/w_gate_up/weight", (96, 160))],
        ids=["q short, k long", "gate short, up long"],
    )
    def test_rejects_a_fused_block_of_the_wrong_size(self, placements, suffix, rows, expect_error):
        """One block is 32 rows short and the next 32 rows long, so together they still fill the parameter.
        Each block must still have the rows the config gives it."""
        rule = next(
            rule
            for rule in _rules(coverage_config(), param_names(_PER_LAYER_FUSED))
            if rule.param == f"Llama/blocks/0/{suffix}"
        )
        headers = HeadersOnly({source: (n, MODEL_HIDDEN) for source, n in zip(rule.sources, rows)})
        with expect_error(RuntimeError, "the config implies"):
            _load_plan(rule, self.param(sum(rows), MODEL_HIDDEN), None, 1, headers)

    @pytest.mark.parametrize("block_rows", [None, (64,)], ids=["none", "too few"])
    def test_a_fused_rule_needs_the_rows_of_every_block(self, block_rows, expect_error):
        """Without each block's rows, the check above would not run for that rule."""
        with expect_error(ValueError, "needs block_rows for each of its 2 sources"):
            _Rule("w_gate_up", ("gate", "up"), block_rows=block_rows)


class TestHelpers:
    @pytest.mark.parametrize(
        "raw,canonical",
        [
            ("model.embed_tokens.weight", "embed_tokens.weight"),
            ("model.wte.weight", "embed_tokens.weight"),
            ("transformer.wte.weight", "embed_tokens.weight"),
            ("lm_head.weight", "lm_head.weight"),
        ],
    )
    def test_canonical_collapses_checkpoint_spellings(self, raw, canonical):
        assert _canonical(raw) == canonical

    def test_pad_grows_the_rows_with_deterministic_noise(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        first, second = _pad_to(arr, (4, 3)), _pad_to(arr, (4, 3))
        assert np.array_equal(first[:2], arr), "checkpoint block moved"
        assert np.array_equal(first, second), "padding differs between loads of one checkpoint"
        assert (first[2:] != 0).all(), "padding is zeros rather than init-like noise"

    def test_pad_never_grows_the_columns(self, expect_error):
        with expect_error(RuntimeError, "only rows are padded"):
            _pad_to(np.zeros((2, 3), np.float32), (4, 6))

    def test_unpermute_matches_the_explicit_definition(self):
        """out[2i] = w[i], out[2i+1] = w[half+i], per head."""
        n_heads, per_head, cols = 3, 6, 2
        w = np.arange(n_heads * per_head * cols, dtype=np.float32).reshape(n_heads * per_head, cols)
        got, half = unpermute_proj_rows(w, n_heads), per_head // 2
        for head in range(n_heads):
            base = head * per_head
            for i in range(half):
                assert np.array_equal(got[base + 2 * i], w[base + i])
                assert np.array_equal(got[base + 2 * i + 1], w[base + half + i])

    def test_unpermute_moves_a_bias_like_a_weight_row(self):
        """Qwen3 unpermutes its 1-D Q/K biases with the same helper."""
        w = np.arange(12, dtype=np.float32).reshape(12, 1)
        assert np.array_equal(unpermute_proj_rows(w[:, 0], 2), unpermute_proj_rows(w, 2)[:, 0])


def save_checkpoint(directory, *shards: dict[str, np.ndarray]) -> None:
    """Each dict as one HF-named shard file."""
    save_file = pytest.importorskip("safetensors.numpy").save_file
    for i, shard in enumerate(shards, 1):
        save_file(shard, str(directory / f"model-{i:05d}-of-{len(shards):05d}.safetensors"))


class TestReadCheckpoint:
    def test_rejects_a_directory_without_safetensors_files(self, tmp_path, expect_error):
        (tmp_path / "config.json").write_text("{}")
        with expect_error(FileNotFoundError, "No .safetensors files found"):
            _Checkpoint(tmp_path)

    def test_rejects_a_quantized_tensor_before_reading_it(self, tmp_path, expect_error):
        """An int8 weight would otherwise load as raw integers, and an FP8 one fail after earlier writes."""
        save_checkpoint(tmp_path, {"model.layers.0.mlp.down_proj.weight": np.zeros((3, 4), np.int8)})
        with _Checkpoint(tmp_path) as checkpoint, expect_error(RuntimeError, "stored as I8"):
            checkpoint.shape("layers.0.mlp.down_proj.weight")

    @pytest.mark.parametrize("dtype", [np.float32, ml_dtypes.bfloat16], ids=["f32", "bf16"])
    def test_reads_a_tensor_in_its_stored_dtype(self, tmp_path, dtype):
        """Real checkpoints are bf16; safetensors resolves it only through ml_dtypes."""
        w = np.arange(12, dtype=np.float32).reshape(3, 4).astype(dtype)
        save_checkpoint(tmp_path, {"model.layers.0.mlp.down_proj.weight": w})
        with _Checkpoint(tmp_path) as checkpoint:
            got = checkpoint["layers.0.mlp.down_proj.weight"]
        assert got.dtype == dtype and got.shape == (3, 4)
        assert np.array_equal(got.astype(np.float32), w.astype(np.float32))

    def test_reshapes_a_1d_gamma_to_one_row(self, tmp_path):
        save_checkpoint(tmp_path, {"model.norm.weight": np.arange(4, dtype=np.float32)})
        with _Checkpoint(tmp_path) as checkpoint:
            assert checkpoint.shape("norm.weight") == (1, 4)
            assert checkpoint["norm.weight"].shape == (1, 4)

    def test_rejects_a_tensor_that_is_neither_1d_nor_2d(self, tmp_path, expect_error):
        save_checkpoint(tmp_path, {"model.norm.weight": np.zeros((2, 2, 2), np.float32)})
        with _Checkpoint(tmp_path) as checkpoint, expect_error(RuntimeError, "expected a 1-D or 2-D tensor"):
            checkpoint.shape("norm.weight")

    def test_rejects_two_spellings_of_one_tensor(self, tmp_path, expect_error):
        zeros = np.zeros((2, 2), np.float32)
        save_checkpoint(tmp_path, {"model.wte.weight": zeros, "embed_tokens.weight": zeros})
        with expect_error(RuntimeError, "collides with another tensor"):
            _Checkpoint(tmp_path)

    def test_rejects_a_tensor_present_in_two_shards(self, tmp_path, expect_error):
        zeros = np.zeros((2, 2), np.float32)
        save_checkpoint(tmp_path, {"norm.weight": zeros}, {"norm.weight": zeros})
        with expect_error(RuntimeError, "collides with another tensor"):
            _Checkpoint(tmp_path)


# ── Coverage: does the rule set match the model? ──

_HF_TOP_LEVEL = {
    "Llama/tok_emb/weight": ("model.embed_tokens.weight",),
    "Llama/fc/weight": ("lm_head.weight",),
    "Llama/ln_fc/gamma": ("model.norm.weight",),
}
_HF_PER_LAYER = {
    "attention_norm/gamma": ("input_layernorm.weight",),
    "mlp_norm/gamma": ("post_attention_layernorm.weight",),
    "attention/qkv_linear/weight": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
    "attention/out_linear/weight": ("self_attn.o_proj.weight",),
    "mlp/w_gate_up/weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
    "mlp/w2/weight": ("mlp.down_proj.weight",),
}
_TOP_LEVEL = tuple(_HF_TOP_LEVEL)
_PER_LAYER_FUSED = tuple(_HF_PER_LAYER)
_PER_LAYER_PRE_FUSION = (
    "attention_norm/gamma",
    "mlp_norm/gamma",
    "attention/q_linear/weight",
    "attention/kv_linear/weight",
    "attention/out_linear/weight",
    "mlp/w1/weight",
    "mlp/w3/weight",
    "mlp/w2/weight",
)


N_LAYERS = 2
N_HEADS = 4
N_KV_HEADS = 2
MODEL_HIDDEN = 128
INTERMEDIATE = 128
VOCAB = 128
SEQ_LEN = 64
MODEL_HEAD_DIM = MODEL_HIDDEN // N_HEADS


def param_names(per_layer):
    return {*_TOP_LEVEL, *(f"Llama/blocks/{i}/{p}" for i in range(N_LAYERS) for p in per_layer)}


def coverage_config(**overrides):
    defaults = dict(
        hidden_size=MODEL_HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LEN,
    )
    return LlamaConfig(**{**defaults, **overrides})


def hf_sources(param: str, tied: bool = False) -> tuple[str, ...]:
    """The HF tensors *param* holds."""
    if tied and param in _TIED_NAMES:
        return _HF_TOP_LEVEL["Llama/tok_emb/weight"]
    if param in _HF_TOP_LEVEL:
        return _HF_TOP_LEVEL[param]
    _, _, layer, suffix = param.split("/", 3)
    return tuple(f"model.layers.{layer}.{source}" for source in _HF_PER_LAYER[suffix])


class TestCoverage:
    """Device-free: ``_rules`` is pure and ``_check_coverage`` takes a set of names."""

    def test_accepts_the_current_model(self):
        config, names = coverage_config(), param_names(_PER_LAYER_FUSED)
        _check_coverage(names, list(_rules(config, names)), frozenset())

    def test_rules_name_the_sources_each_parameter_holds(self):
        config, names = coverage_config(), param_names(_PER_LAYER_FUSED)
        rules = list(_rules(config, names))
        assert {r.param: r.sources for r in rules} == {n: tuple(map(_canonical, hf_sources(n))) for n in names}
        transformed = {r.param for r in rules if r.transform is not None}
        assert transformed == {n for n in names if n.endswith("qkv_linear/weight")}, "only qkv is unpermuted"

    def test_rejects_the_pre_fusion_model_naming_both_directions(self, expect_error):
        """The break this loader was fixed for: the model fused q/k/v and gate/up while the
        rules still named q_linear/kv_linear/w1/w3."""
        config, names = coverage_config(), param_names(_PER_LAYER_PRE_FUSION)
        with expect_error(RuntimeError, "disagree about its parameters") as excinfo:
            _check_coverage(names, list(_rules(config, names)), frozenset())
        message = str(excinfo.value)
        assert "no rule feeds       Llama/blocks/0/attention/q_linear/weight" in message
        assert "no such parameter   Llama/blocks/0/attention/qkv_linear/weight" in message

    def test_attention_biases_are_declared_init_only(self):
        """HF Llama ships no attention biases, so a biased model must still be loadable."""
        config = coverage_config(attention_bias=True)
        names = param_names(_PER_LAYER_FUSED) | {
            f"Llama/blocks/{i}/attention/{linear}/bias"
            for i in range(N_LAYERS)
            for linear in ("qkv_linear", "out_linear")
        }
        _check_coverage(names, list(_rules(config, names)), frozenset())

    def test_rejects_a_checkpoint_that_ships_a_bias_the_model_has(self, expect_error):
        config = coverage_config(attention_bias=True)
        names = param_names(_PER_LAYER_FUSED) | {"Llama/blocks/0/attention/qkv_linear/bias"}
        with expect_error(RuntimeError, "bias shipped for    Llama/blocks/0/attention/qkv_linear/bias"):
            _check_coverage(names, list(_rules(config, names)), frozenset({"layers.0.self_attn.q_proj.bias"}))

    def test_a_bias_for_another_linear_keeps_the_exemption(self):
        config = coverage_config(attention_bias=True)
        names = param_names(_PER_LAYER_FUSED) | {"Llama/blocks/0/attention/qkv_linear/bias"}
        _check_coverage(names, list(_rules(config, names)), frozenset({"lm_head.bias"}))

    def test_rejects_two_rules_for_one_parameter(self, expect_error):
        """A set of targets would hide the duplicate, and the load loop would assign the parameter twice."""
        config, names = coverage_config(), param_names(_PER_LAYER_FUSED)
        rules = list(_rules(config, names))
        twice = next(rule for rule in rules if rule.param == "Llama/ln_fc/gamma")
        with expect_error(RuntimeError, r"fed by 2 rules\s+Llama/ln_fc/gamma"):
            _check_coverage(names, rules + [twice], frozenset())

    @pytest.mark.parametrize("survivor", ["Llama/fc/weight", "Llama/tok_emb/weight"])
    def test_tying_feeds_whichever_name_survived(self, survivor):
        """Which of the two names survives dedup is a module-traversal detail, not a contract."""
        config = coverage_config(weight_tying=WeightTyingType.Enabled)
        names = (param_names(_PER_LAYER_FUSED) - set(_TIED_NAMES)) | {survivor}
        targets = [rule.param for rule in _rules(config, names)]
        assert targets.count(survivor) == 1, "tied embedding must be written once"
        _check_coverage(names, list(_rules(config, names)), frozenset())

    def test_tying_rejects_two_separate_parameters(self, expect_error):
        """Both names present means unshared tensors: the rule feeds one, coverage flags the other."""
        config, names = coverage_config(weight_tying=WeightTyingType.Enabled), param_names(_PER_LAYER_FUSED)
        with expect_error(RuntimeError, "no rule feeds       Llama/tok_emb/weight"):
            _check_coverage(names, list(_rules(config, names)), frozenset())


class TestSources:
    """Runs without a device. ``_check_sources`` reads tensor data
    only to compare a tied checkpoint's ``lm_head.weight`` with its embedding."""

    @staticmethod
    def rules(tying: WeightTyingType):
        names = param_names(_PER_LAYER_FUSED)
        if tying == WeightTyingType.Enabled:
            names -= {"Llama/fc/weight"}
        return list(_rules(coverage_config(weight_tying=tying), names))

    @staticmethod
    def sources_of(rules):
        return {source for rule in rules for source in rule.sources}

    def test_a_missing_source_names_its_parameter(self, expect_error):
        rules = self.rules(WeightTyingType.Disabled)
        headers = HeadersOnly(dict.fromkeys(self.sources_of(rules) - {"layers.0.mlp.up_proj.weight"}, (1, 1)))
        with expect_error(RuntimeError, r"blocks/0/mlp/w_gate_up/weight: the checkpoint has no .*up_proj.weight"):
            _check_sources(rules, headers)

    def test_tied_checkpoint_on_an_untied_model_says_to_tie(self, expect_error):
        rules = self.rules(WeightTyingType.Disabled)
        with expect_error(RuntimeError, "load it with weight_tying=Enabled"):
            _check_sources(rules, HeadersOnly(dict.fromkeys(self.sources_of(rules) - {"lm_head.weight"}, (1, 1))))

    @staticmethod
    def write_tied(directory, rules, lm_head_offset: float) -> _Checkpoint:
        """Write every source as zeros, a random embedding, and ``lm_head = embedding + lm_head_offset``."""
        tensors = {name: np.zeros((2, 2), np.float32) for name in TestSources.sources_of(rules)}
        tensors["embed_tokens.weight"] = np.random.default_rng(3).standard_normal((4, 2)).astype(np.float32)
        tensors["lm_head.weight"] = tensors["embed_tokens.weight"] + lm_head_offset
        save_checkpoint(directory, tensors)
        return _Checkpoint(directory)

    def test_untied_checkpoint_on_a_tied_model_says_to_untie(self, tmp_path, expect_error):
        rules = self.rules(WeightTyingType.Enabled)
        with (
            self.write_tied(tmp_path, rules, 1.0) as checkpoint,
            expect_error(RuntimeError, "load it with weight_tying=Disabled"),
        ):
            _check_sources(rules, checkpoint)

    def test_a_tied_checkpoint_may_keep_a_copy_of_its_embedding_as_lm_head(self, tmp_path):
        """Some tied exports include lm_head.weight as a copy of the embedding; HF ignores it, and so do we."""
        rules = self.rules(WeightTyingType.Enabled)
        with self.write_tied(tmp_path, rules, 0.0) as checkpoint:
            _check_sources(rules, checkpoint)


# ── End-to-end: a synthetic HF checkpoint through the real loader ──


def e2e_config(use_tp: bool, placement: EmbeddingPlacement, tying=WeightTyingType.Disabled, **overrides):
    return coverage_config(use_tp=use_tp, embedding_placement=placement, weight_tying=tying, **overrides)


def hf_tensors(vocab: int = VOCAB, dtype=np.float32, tied: bool = False) -> dict[str, np.ndarray]:
    """A synthetic HF Llama checkpoint; real ones are bf16, so *dtype* covers that read path."""
    rng = np.random.default_rng(99)

    def w(rows, cols):
        return rng.standard_normal((rows, cols)).astype(dtype)

    def gamma():
        return rng.standard_normal(MODEL_HIDDEN).astype(dtype)

    tensors = {
        "model.embed_tokens.weight": w(vocab, MODEL_HIDDEN),
        "lm_head.weight": w(vocab, MODEL_HIDDEN),
        "model.norm.weight": gamma(),
    }
    if tied:
        del tensors["lm_head.weight"]
    for layer in range(N_LAYERS):
        pfx = f"model.layers.{layer}"
        tensors.update(
            {
                f"{pfx}.input_layernorm.weight": gamma(),
                f"{pfx}.post_attention_layernorm.weight": gamma(),
                f"{pfx}.self_attn.q_proj.weight": w(N_HEADS * MODEL_HEAD_DIM, MODEL_HIDDEN),
                f"{pfx}.self_attn.k_proj.weight": w(N_KV_HEADS * MODEL_HEAD_DIM, MODEL_HIDDEN),
                f"{pfx}.self_attn.v_proj.weight": w(N_KV_HEADS * MODEL_HEAD_DIM, MODEL_HIDDEN),
                f"{pfx}.self_attn.o_proj.weight": w(MODEL_HIDDEN, MODEL_HIDDEN),
                f"{pfx}.mlp.gate_proj.weight": w(INTERMEDIATE, MODEL_HIDDEN),
                f"{pfx}.mlp.up_proj.weight": w(INTERMEDIATE, MODEL_HIDDEN),
                f"{pfx}.mlp.down_proj.weight": w(MODEL_HIDDEN, INTERMEDIATE),
            }
        )
    return tensors


def write_hf_checkpoint(directory, tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Save *tensors* as two shard files, alternating tensors so each fused group spans both files."""
    items = list(tensors.items())
    save_checkpoint(directory, dict(items[0::2]), dict(items[1::2]))
    return tensors


def as_stored(param, array: np.ndarray) -> np.ndarray:
    """*array* as *param* stores it, so a read-back comparison can be exact: an fp32 parameter
    keeps the checkpoint's values, a bf16 one rounds them."""
    if param.dtype() == ttnn.DataType.FLOAT32:
        return array.astype(np.float32)
    return array.astype(ml_dtypes.bfloat16).astype(np.float32)


def gather(tensor, shard_dim: int | None) -> np.ndarray:
    """A mapper-placed *tensor* as one float32 host array. ``Sharding.gather`` follows whatever placement
    the tensor has, so the assert is what catches a weight sharded on the wrong dim."""
    sharding = ttml.Sharding.from_tensor(tensor)
    if shard_dim is None:
        assert sharding.is_fully_replicated, f"expected a replicated tensor, got {sharding.placements}"
    else:
        placement = sharding.placements[ttml.mesh().axis_index("tp")]
        assert (
            isinstance(placement, ttnn.PlacementShard) and placement.dim == shard_dim
        ), f"expected a shard on dim {shard_dim} over 'tp', got {sharding.placements}"
    return sharding.gather(tensor).astype(np.float32)


def store_as(model, dtype) -> None:
    """Store every parameter as *dtype*, as a run that trains in that precision does. New parameters are bf16."""
    precision = (
        ttml.autograd.PreferredPrecision.FULL
        if dtype == ttnn.DataType.FLOAT32
        else ttml.autograd.PreferredPrecision.HALF
    )
    for param in model.parameters().values():
        param.set_value(param.get_value(precision))


def read_param(params, name: str, shard_dim: int | None) -> np.ndarray:
    """A parameter as a 2-D ``[out_features, in_features]`` array."""
    full = gather(params[name], shard_dim)
    return full.reshape(full.shape[-2], full.shape[-1])


def tp_shard_dim(param: str, placement: EmbeddingPlacement) -> int | None:
    if param.endswith(("qkv_linear/weight", "w_gate_up/weight", "Llama/fc/weight")):
        return ROW_DIM
    if param.endswith(("out_linear/weight", "w2/weight")):
        return COL_DIM
    if param == "Llama/tok_emb/weight":
        return {
            EmbeddingPlacement.Replicated: None,
            EmbeddingPlacement.VocabParallel: ROW_DIM,
            EmbeddingPlacement.FeatureParallel: COL_DIM,
        }[placement]
    return None


def expected_value(param: str, hf: dict[str, np.ndarray], tp_size: int, tied: bool) -> np.ndarray:
    """*param*'s full ``[rows, cols]`` value, laid out for *tp_size* ranks."""
    sources = [hf[name] for name in hf_sources(param, tied)]
    if param.endswith("qkv_linear/weight"):
        q, k, v = sources
        return fuse_qkv(q, k, v, N_HEADS, N_KV_HEADS, tp_size)
    if param.endswith("w_gate_up/weight"):
        gate, up = sources
        return fuse_gate_up(gate, up, tp_size)
    (single,) = sources
    return single.reshape(1, -1) if single.ndim == 1 else single


def assert_every_parameter_loaded(model, hf, tp_size=1, placement=EmbeddingPlacement.Replicated):
    """Every parameter holds its HF sources bit-exactly, in the layout its placement demands. A
    tile-padded parameter holds them in its leading rows, with noise rather than zeros below."""
    params = model.parameters()
    tied = model.config.weight_tying == WeightTyingType.Enabled
    for name in params:
        expected = as_stored(params[name], expected_value(name, hf, tp_size, tied))
        got = read_param(params, name, tp_shard_dim(name, placement) if tp_size > 1 else None)
        rows = expected.shape[0]
        assert got.shape[0] >= rows and got.shape[1] == expected.shape[1], f"{name}: {got.shape} vs {expected.shape}"
        assert np.array_equal(got[:rows], expected), f"{name}: does not hold its checkpoint sources"
        if got.shape[0] > rows:
            assert got[rows:].any(), f"{name}: padding is all zeros rather than init-like noise"


TYINGS = [WeightTyingType.Disabled, WeightTyingType.Enabled]


@pytest.mark.requires_device
class TestLoadIntoModel:
    @pytest.mark.parametrize("dtype", [np.float32, ml_dtypes.bfloat16], ids=["f32", "bf16"])
    def test_reports_a_clean_load(self, tmp_path, dtype):
        write_hf_checkpoint(tmp_path, hf_tensors(dtype=dtype))
        model = Llama(e2e_config(False, EmbeddingPlacement.Replicated))
        result = load_from_safetensors(model, tmp_path)

        assert sorted(result.loaded) == sorted(model.parameters())
        assert result.unused == () and result.left_at_init == (), result

    def test_leaves_biases_at_init_and_ignores_rotary_buffers(self, tmp_path):
        """HF Llama ships no biases, so a biased model keeps their init values; inv_freq is derived at runtime."""
        hf = hf_tensors()
        hf["model.layers.0.self_attn.rotary_emb.inv_freq"] = np.ones(MODEL_HEAD_DIM // 2, np.float32)
        write_hf_checkpoint(tmp_path, hf)
        model = Llama(e2e_config(False, EmbeddingPlacement.Replicated, attention_bias=True))
        biases = sorted(name for name in model.parameters() if name.endswith("/bias"))
        assert biases, "premise: the model has attention biases"
        before = {name: model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32) for name in biases}

        result = load_from_safetensors(model, tmp_path)

        assert result.left_at_init == tuple(biases)
        assert result.unused == ()
        for name in biases:
            assert np.array_equal(model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32), before[name]), name
        weights = {name: param for name, param in model.parameters().items() if not name.endswith("/bias")}
        assert sorted(result.loaded) == sorted(weights)

    def test_rejects_a_checkpoint_missing_a_fused_source(self, tmp_path, expect_error):
        """A fused parameter needs all of its sources; a partial group must not load silently."""
        save_checkpoint(tmp_path, {k: v for k, v in hf_tensors().items() if "v_proj" not in k})
        with expect_error(RuntimeError, "the checkpoint has no"):
            load_from_safetensors(Llama(e2e_config(False, EmbeddingPlacement.Replicated)), tmp_path)

    def test_a_failed_check_leaves_the_model_untouched(self, tmp_path, expect_error):
        """A bad tensor in the last layer must fail the load before any earlier layer is written."""
        hf = hf_tensors()
        hf[f"model.layers.{N_LAYERS - 1}.mlp.down_proj.weight"] = np.zeros(
            (MODEL_HIDDEN, INTERMEDIATE // 2), np.float32
        )
        write_hf_checkpoint(tmp_path, hf)
        model = Llama(e2e_config(False, EmbeddingPlacement.Replicated))
        before = {name: param.to_numpy(ttnn.DataType.FLOAT32) for name, param in model.parameters().items()}

        with expect_error(RuntimeError, "does not match the config-implied shape"):
            load_from_safetensors(model, tmp_path)
        for name, param in model.parameters().items():
            assert np.array_equal(param.to_numpy(ttnn.DataType.FLOAT32), before[name]), f"{name} was assigned"

    @pytest.mark.parametrize(
        "short,long",
        [("self_attn.q_proj", "self_attn.k_proj"), ("mlp.gate_proj", "mlp.up_proj")],
        ids=["q short, k long", "gate short, up long"],
    )
    def test_rejects_a_fused_block_of_the_wrong_size(self, tmp_path, short, long, expect_error):
        """One block is a head's width short and the next as much long, so together they still fill the fused parameter,
        and the RoPE unpermute still splits q and k into heads. Without a check on each block,
        the long block's first rows would load where the short one's last rows belong."""
        hf = hf_tensors()
        pfx = "model.layers.0"
        short_block, long_block = hf[f"{pfx}.{short}.weight"], hf[f"{pfx}.{long}.weight"]
        hf[f"{pfx}.{short}.weight"] = short_block[:-MODEL_HEAD_DIM]
        hf[f"{pfx}.{long}.weight"] = np.concatenate([long_block, short_block[-MODEL_HEAD_DIM:]])
        write_hf_checkpoint(tmp_path, hf)
        with expect_error(RuntimeError, "the config implies"):
            load_from_safetensors(Llama(e2e_config(False, EmbeddingPlacement.Replicated)), tmp_path)

    @pytest.mark.parametrize("tying", TYINGS, ids=lambda v: v.name)
    def test_every_parameter_holds_its_sources(self, tmp_path, tying):
        hf = write_hf_checkpoint(tmp_path, hf_tensors(tied=tying == WeightTyingType.Enabled))
        model = Llama(e2e_config(False, EmbeddingPlacement.Replicated, tying))
        load_from_safetensors(model, tmp_path)
        assert_every_parameter_loaded(model, hf)

    @pytest.mark.parametrize(
        "checkpoint_dtype", [np.float32, ml_dtypes.bfloat16, np.float16], ids=["f32 file", "bf16 file", "f16 file"]
    )
    @pytest.mark.parametrize("dtype", [ttnn.DataType.FLOAT32, ttnn.DataType.BFLOAT16], ids=["f32 model", "bf16 model"])
    def test_keeps_the_models_dtype(self, tmp_path, dtype, checkpoint_dtype):
        """The model decides the precision and the checkpoint supplies the values: an fp32 parameter
        holds the checkpoint's values exactly, a bf16 one their rounding."""
        hf = write_hf_checkpoint(tmp_path, hf_tensors(dtype=checkpoint_dtype))
        model = Llama(e2e_config(False, EmbeddingPlacement.Replicated))
        store_as(model, dtype)
        load_from_safetensors(model, tmp_path)
        for name, param in model.parameters().items():
            assert param.dtype() == dtype, f"{name}: stored as {param.dtype()}"
        assert_every_parameter_loaded(model, hf)

    def test_a_tied_checkpoint_may_keep_a_copy_of_its_embedding_as_lm_head(self, tmp_path):
        hf = hf_tensors()
        hf["lm_head.weight"] = hf["model.embed_tokens.weight"].copy()
        write_hf_checkpoint(tmp_path, hf)
        model = Llama(e2e_config(False, EmbeddingPlacement.Replicated, WeightTyingType.Enabled))
        assert load_from_safetensors(model, tmp_path).unused == ("lm_head.weight",)
        assert_every_parameter_loaded(model, hf)

    def test_pads_the_vocab_the_model_rounds_up_to_a_tile(self, tmp_path):
        vocab = VOCAB - 8
        hf = write_hf_checkpoint(tmp_path, hf_tensors(vocab=vocab))
        model = Llama(e2e_config(False, EmbeddingPlacement.Replicated, vocab_size=vocab))
        assert model.padded_vocab_size > vocab, "premise: the model rounds the vocab up"
        load_from_safetensors(model, tmp_path)
        assert_every_parameter_loaded(model, hf)

    @pytest.mark.parametrize("vocab", [VOCAB - 8, VOCAB + 8], ids=["shorter", "longer"])
    def test_rejects_a_checkpoint_whose_vocab_disagrees_with_the_config(self, tmp_path, vocab, expect_error):
        """A shorter vocab would leave noise for real tokens; main cropped a longer one."""
        write_hf_checkpoint(tmp_path, hf_tensors(vocab=vocab))
        with expect_error(RuntimeError, "does not match the config-implied shape"):
            load_from_safetensors(Llama(e2e_config(False, EmbeddingPlacement.Replicated)), tmp_path)


# ── Against HuggingFace ──

# Over six seeds, a correct load measured at most 13 ULP at the peak and 4 at the p99
# (5.5 with llama3 RoPE). A load that skips the RoPE unpermute, swaps gate and up, or misreads rope_theta
# measured at least 62 and 41; one that ignores llama3 scaling, at least 198 and 134.
HF_MAX_ULP = 32.0
HF_MAX_ULP_P99 = 8.0


# Llama 3's RoPE scaling, shrunk so the scaled bands rotate within SEQ_LEN positions: wavelengths
# past original_max_position_embeddings / low_freq_factor = 16 are stretched 8x, and 16 / high_freq_factor..16
# are blended.
LLAMA3_ROPE = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 16,
}


def hf_llama(directory, tied: bool, rope: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Save a small random ``LlamaForCausalLM`` into *directory* in bf16, as real checkpoints ship.
    Return a prompt and HF's float32 logits for it, computed from those same bf16 weights.
    *rope* adds a RoPE scaling, e.g. ``LLAMA3_ROPE``."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        hidden_size=MODEL_HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LEN,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0, **(rope or {})},
        tie_word_embeddings=tied,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(torch.rand_like(param) + 0.5)  # distinct gains, so swapping two norms shows
            else:
                # HF's 0.02 init leaves attention near uniform, where skipping the RoPE unpermute passes.
                param.copy_(torch.randn_like(param) * MODEL_HIDDEN**-0.5)
    model.to(torch.bfloat16).save_pretrained(directory)
    model.float()  # exact: every bf16 value is a float32

    ids = np.random.default_rng(7).integers(0, VOCAB, (1, SEQ_LEN))
    with torch.no_grad():
        return ids, model(torch.from_numpy(ids)).logits[0].numpy()


def ttml_logits(model: Llama, ids: np.ndarray) -> np.ndarray:
    """*model*'s ``[seq, vocab]`` logits for *ids*."""
    logits = model(
        ttml.autograd.Tensor.from_numpy(
            ids.reshape(1, 1, 1, -1).astype(np.uint32), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
        ),
        build_causal_mask(SEQ_LEN, device=True),
    )
    composer = None
    if model.config.use_tp:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, COL_DIM)
    values = logits.to_numpy(ttnn.DataType.FLOAT32, composer=composer)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return values.reshape(SEQ_LEN, -1)[:, :VOCAB]


def assert_logits_match_hf(directory, tied: bool, rope: dict | None = None, **config_overrides) -> None:
    """Build the model from HF's config.json and load its checkpoint, as the examples do, then compare logits."""
    ids, expected = hf_llama(directory, tied, rope)
    model = Llama(LlamaConfig.from_hf(directory, max_position_embeddings=SEQ_LEN, **config_overrides))
    load_from_safetensors(model, directory)
    model.eval()
    # Logits cross zero and their error is absolute, so every element is measured in ULP at the peak.
    assert_within_bf16_ulp(
        ttml_logits(model, ids), expected, "logits vs HF", HF_MAX_ULP, HF_MAX_ULP_P99, p99_floor_binades=0
    )


@pytest.mark.requires_device
class TestMatchesHuggingFace:
    @pytest.mark.parametrize("rope", [None, LLAMA3_ROPE], ids=["default rope", "llama3 rope"])
    @pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
    def test_logits(self, tmp_path, tied, rope):
        assert_logits_match_hf(tmp_path, tied, rope)

    def test_sees_llama3_scaling_ignored(self, tmp_path, expect_error):
        """The llama3 case above can tell scaled RoPE from plain RoPE."""
        with expect_error(AssertionError, "logits vs HF: ulp="):
            assert_logits_match_hf(tmp_path, tied=False, rope=LLAMA3_ROPE, rope_scaling=LlamaRopeScalingConfig())

    def test_sees_q_and_k_left_in_hf_row_order(self, tmp_path, monkeypatch, expect_error):
        """The end-to-end oracle shares this bug with the loader, so only HF can see it."""
        monkeypatch.setattr(safetensors_loader, "unpermute_proj_rows", lambda rows, n_heads: rows)
        with expect_error(AssertionError, "logits vs HF: ulp="):
            assert_logits_match_hf(tmp_path, tied=False)


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh")
class TestLoadIntoTensorParallelModel:
    @pytest.mark.parametrize(
        "placement,tying",
        [
            (EmbeddingPlacement.Replicated, WeightTyingType.Disabled),
            (EmbeddingPlacement.VocabParallel, WeightTyingType.Disabled),
            (EmbeddingPlacement.VocabParallel, WeightTyingType.Enabled),
            (EmbeddingPlacement.FeatureParallel, WeightTyingType.Disabled),
        ],
        ids=lambda v: v.name,
    )
    def test_tensor_parallel_layout(self, tmp_path, placement, tying):
        hf = write_hf_checkpoint(tmp_path, hf_tensors(tied=tying == WeightTyingType.Enabled))
        model = Llama(e2e_config(True, placement, tying))
        load_from_safetensors(model, tmp_path)
        assert_every_parameter_loaded(model, hf, ttml.mesh().axis_size("tp"), placement)

    def test_pads_the_vocab_the_model_rounds_up_to_a_tile(self, tmp_path):
        """The one end-to-end vocab that is not tile-aligned: 120 rounds up to 128, and the pad rows shard too."""
        vocab = VOCAB - 8
        hf = write_hf_checkpoint(tmp_path, hf_tensors(vocab=vocab))
        model = Llama(e2e_config(True, EmbeddingPlacement.VocabParallel, vocab_size=vocab))
        assert model.padded_vocab_size > vocab, "premise: the model rounds the vocab up"
        load_from_safetensors(model, tmp_path)
        assert_every_parameter_loaded(model, hf, ttml.mesh().axis_size("tp"), EmbeddingPlacement.VocabParallel)

    def test_keeps_an_fp32_models_values(self, tmp_path):
        """The fused blocks are interleaved and sharded in the parameter's dtype, not rounded to bf16 on the way."""
        hf = write_hf_checkpoint(tmp_path, hf_tensors())
        model = Llama(e2e_config(True, EmbeddingPlacement.VocabParallel))
        store_as(model, ttnn.DataType.FLOAT32)
        load_from_safetensors(model, tmp_path)
        for name, param in model.parameters().items():
            assert param.dtype() == ttnn.DataType.FLOAT32, f"{name}: stored as {param.dtype()}"
        assert_every_parameter_loaded(model, hf, ttml.mesh().axis_size("tp"), EmbeddingPlacement.VocabParallel)

    def test_logits_match_hugging_face(self, tmp_path):
        """Tied and vocab-parallel, as LoRA loads Llama 3.2 1B, with the fused blocks interleaved per device."""
        assert_logits_match_hf(tmp_path, tied=True, use_tp=True, embedding_placement=EmbeddingPlacement.VocabParallel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
