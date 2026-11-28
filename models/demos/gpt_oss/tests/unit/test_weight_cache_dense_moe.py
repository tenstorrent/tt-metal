# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of ModelArgs.weight_cache_is_complete for the batched dense MoE weights (dense_moe=True).

A marker that records the dense MoE weight set is not enough: the dense_decode_* cache files of every layer must be
there too, else the HF weights load so the build writes them (CI's shared cache, 2026-11-24: a marker with the set and
no files made the batched model fail in ttnn.as_tensor(None, ...)). No device is needed.
"""

import json

from models.demos.gpt_oss.tt.model_config import ModelArgs

N_LAYERS = 2


def _args(monkeypatch, cache_path):
    args = ModelArgs.__new__(ModelArgs)  # no mesh device or HF config: only what the check reads
    args.model_name = "gpt-oss-20b"
    args.n_layers = N_LAYERS
    monkeypatch.setattr(ModelArgs, "weight_cache_path", lambda self, dtype: cache_path)
    monkeypatch.setattr(ModelArgs, "_routed_prefill_weights_version", lambda self: None)
    monkeypatch.delenv("GPT_OSS_FORCE_MODEL_LOAD", raising=False)
    for name in ModelArgs.DENSE_SETTING_VARS:  # the GPTQ dense weights are a marker setting: none here
        monkeypatch.delenv(name, raising=False)
    return args


def _marker(cache_path, dense_moe):
    (cache_path / ModelArgs.WEIGHT_CACHE_MARKER).write_text(
        json.dumps(
            {
                "format_version": ModelArgs.WEIGHT_CACHE_FORMAT_VERSION,
                "model_name": "gpt-oss-20b",
                "n_layers": N_LAYERS,
                "fused_decode_weights": False,
                "dense_moe_decode_weights": ModelArgs.DENSE_MOE_DECODE_WEIGHTS_VERSION if dense_moe else False,
                "routed_prefill_weights": False,
            }
        )
    )
    (cache_path / "embedding_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin").write_bytes(b"")


def _dense_files(cache_path, layers, names=ModelArgs.DENSE_MOE_CACHED_WEIGHTS):
    for i in layers:
        experts = cache_path / f"model.layers.{i}" / "mlp" / "experts"
        experts.mkdir(parents=True, exist_ok=True)
        for name in names:
            (experts / f"{name}_dtype_BFLOAT4_B_layout_TILE.tensorbin").write_bytes(b"")


def test_marker_without_dense_files_is_incomplete(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    _marker(tmp_path, dense_moe=True)
    assert args.weight_cache_is_complete(None, dense_moe=False)  # models without the dense MoE are unaffected
    assert not args.weight_cache_is_complete(None, dense_moe=True)


def test_dense_files_on_every_layer_are_complete(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    _marker(tmp_path, dense_moe=True)
    _dense_files(tmp_path, range(N_LAYERS))
    assert args.weight_cache_is_complete(None, dense_moe=True)


def test_one_layer_or_weight_missing_is_incomplete(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    _marker(tmp_path, dense_moe=True)
    _dense_files(tmp_path, range(N_LAYERS - 1))
    assert not args.weight_cache_is_complete(None, dense_moe=True)
    _dense_files(tmp_path, [N_LAYERS - 1], names=("dense_decode_gate", "dense_decode_up"))
    assert not args.weight_cache_is_complete(None, dense_moe=True)
    _dense_files(tmp_path, [N_LAYERS - 1])
    assert args.weight_cache_is_complete(None, dense_moe=True)


def test_bias_files_do_not_count_as_weights(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    _marker(tmp_path, dense_moe=True)
    _dense_files(tmp_path, range(N_LAYERS), names=("dense_decode_gate_bias_t32", "dense_decode_up_bias_t32"))
    _dense_files(tmp_path, range(N_LAYERS), names=("dense_decode_down",))
    assert not args.weight_cache_is_complete(None, dense_moe=True)


def test_marker_without_the_dense_set_is_still_incomplete(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    _marker(tmp_path, dense_moe=False)
    _dense_files(tmp_path, range(N_LAYERS))
    assert not args.weight_cache_is_complete(None, dense_moe=True)
