# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.common.models.mistral_7b import model as mistral_model


def test_prefill_runtime_slice_and_index_override_full_hidden_state_return(monkeypatch):
    calls = []
    hidden = SimpleNamespace(shape=(1, 1, 128, 4096), dtype=mistral_model.ttnn.bfloat16)
    sliced = SimpleNamespace(dtype=mistral_model.ttnn.bfloat16)
    selected = object()
    selected_4d = object()
    logits = object()
    last_token_slice = (object(), object())
    last_token_index = object()
    model = SimpleNamespace(
        layers=[],
        num_devices=1,
        _last_tile_logits=lambda value: calls.append(("last_tile_logits", value)) or logits,
    )

    monkeypatch.setattr(
        mistral_model.ttnn,
        "slice",
        lambda value, start, end, **kwargs: calls.append(("slice", value, start, end, kwargs)) or sliced,
    )
    monkeypatch.setattr(
        mistral_model.ttnn,
        "embedding",
        lambda index, value, **kwargs: calls.append(("embedding", index, value, kwargs)) or selected,
    )
    monkeypatch.setattr(
        mistral_model.ttnn,
        "unsqueeze_to_4D",
        lambda value: calls.append(("unsqueeze_to_4D", value)) or selected_4d,
    )
    monkeypatch.setattr(mistral_model.ttnn, "deallocate", lambda value: calls.append(("deallocate", value)))

    result = mistral_model.Mistral7B.prefill_forward(
        model,
        hidden,
        rot_mats=(object(), object()),
        get_last_token=-1,
        last_token_slice=last_token_slice,
        last_token_index=last_token_index,
    )

    assert result is logits
    assert any(call[0] == "slice" and call[1] is hidden for call in calls)
    assert any(call[0] == "embedding" and call[1] is last_token_index for call in calls)
    assert calls[-1] == ("last_tile_logits", selected_4d)


def test_prefill_runtime_index_requires_runtime_slice(expect_error):
    model = SimpleNamespace(layers=[])

    with expect_error(ValueError, "last_token_index is required with a runtime last_token_slice"):
        mistral_model.Mistral7B.prefill_forward(
            model,
            object(),
            rot_mats=(object(), object()),
            get_last_token=-1,
            last_token_index=object(),
        )
