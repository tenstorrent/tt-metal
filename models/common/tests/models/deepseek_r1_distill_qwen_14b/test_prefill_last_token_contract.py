# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.common.models.deepseek_r1_distill_qwen_14b import model as qwen_model


def test_prefill_runtime_slice_and_index_override_full_hidden_state_return(monkeypatch):
    calls = []
    hidden = SimpleNamespace(shape=(1, 1, 128, 640), dtype=qwen_model.ttnn.bfloat16)
    sliced = SimpleNamespace(dtype=qwen_model.ttnn.bfloat16)
    selected = object()
    selected_4d = object()
    logits = object()
    slice_start = object()
    slice_end = object()
    last_token_index = object()
    model = SimpleNamespace(
        layers=[],
        _last_tile_logits=lambda value: calls.append(("last_tile_logits", value)) or logits,
    )

    monkeypatch.setattr(
        qwen_model.ttnn,
        "slice",
        lambda value, start, end, **kwargs: calls.append(("slice", value, start, end, kwargs)) or sliced,
    )
    monkeypatch.setattr(
        qwen_model.ttnn,
        "embedding",
        lambda index, value, **kwargs: calls.append(("embedding", index, value, kwargs)) or selected,
    )
    monkeypatch.setattr(
        qwen_model.ttnn,
        "unsqueeze_to_4D",
        lambda value: calls.append(("unsqueeze_to_4D", value)) or selected_4d,
    )
    monkeypatch.setattr(qwen_model.ttnn, "deallocate", lambda value: calls.append(("deallocate", value)))

    result = qwen_model.DeepSeekR1Qwen14B.prefill_forward(
        model,
        hidden,
        rot_mats=(object(), object()),
        get_last_token=-1,
        last_token_slice=(slice_start, slice_end),
        last_token_index=last_token_index,
    )

    assert result is logits
    assert calls == [
        ("slice", hidden, slice_start, slice_end, {"slice_dim": 2, "num_devices": 4}),
        ("deallocate", hidden),
        ("embedding", last_token_index, sliced, {"layout": qwen_model.ttnn.TILE_LAYOUT}),
        ("unsqueeze_to_4D", selected),
        ("deallocate", sliced),
        ("last_tile_logits", selected_4d),
    ]


def test_prefill_runtime_index_requires_runtime_slice(expect_error):
    model = SimpleNamespace(layers=[])

    with expect_error(ValueError, "last_token_index is required with a runtime last_token_slice"):
        qwen_model.DeepSeekR1Qwen14B.prefill_forward(
            model,
            object(),
            rot_mats=(object(), object()),
            get_last_token=-1,
            last_token_index=object(),
        )
