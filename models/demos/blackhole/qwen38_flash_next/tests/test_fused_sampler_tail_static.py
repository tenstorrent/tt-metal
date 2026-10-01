# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static pins of the one-program sampler (``ttnn/fused/sampler_tail``): the registry entry, the kernel's arithmetic
sites (the composite's two rounding points, the lane order, the literal 128-lane counts), the constants' layout, the
device policy's presence admission and the history image."""

import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler as ds
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    SAMPLING_CANDIDATE_ROW_SHAPE,
    TOKEN_ROW_SHAPE,
    VOCAB_SIZE,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import sampler_tail as st
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38SamplingParameters

KERNEL = (fp.REPO_ROOT / st.KERNEL).read_text(encoding="utf-8")


def test_registry_entry_is_bitwise_and_on_by_default():
    entry = fused.kernel(st.NAME)
    assert entry.tolerance == fused.BITWISE and entry.fused is st.sampler_tail
    assert entry.composed is st.sample_on_device_composed and entry.default_on
    assert "sampler_tail" in fused.__all__


def test_pinned_constants_match_the_model_modules():
    assert st.LANES == ds.LANES and st.MAX_TOP_K == ds.MAX_TOP_K and st.TABLE_SIZE == ds.TABLE_SIZE
    assert st.VOCAB_SIZE == VOCAB_SIZE and st.ROW_LANES == SAMPLING_CANDIDATE_ROW_SHAPE[-1]
    assert st.TOKEN_ROW_SHAPE == TOKEN_ROW_SHAPE and st.HIST_WORDS * 32 >= VOCAB_SIZE > (st.HIST_WORDS - 1) * 32
    assert st.stage_bytes(1) <= 3 * 4096 and st.stage_bytes(st.MAX_ROWS) <= 4 * 4096


def test_kernel_reproduces_the_composite_arithmetic():
    # the lane order: value descending through the sign-magnitude key (-0.0 as +0.0), global id ascending on ties
    assert "bits = (bits & 0x7FFFFFFFu) ? bits : 0u;" in KERNEL
    assert "keys[j] > best_key || (keys[j] == best_key && ids[j] < best_id)" in KERNEL
    # the table index: the fp32 subtract, the exact power-of-two scaling, floor and clamp
    assert "as_float(sorted_values[k]) - s_max;" in KERNEL and "below * -1024.0f;" in KERNEL
    assert "index = TABLE_SIZE - 1;" in KERNEL
    # the two RNE sites and the 128-lane counts the composite takes
    assert "const float tau = top_p * static_cast<float>(total_top_k);" in KERNEL
    assert "const float theta = as_float(uniform_words[r]) * static_cast<float>(total_kept);" in KERNEL
    assert KERNEL.count("for (uint32_t j = 0; j < LANES; ++j) {") >= 3
    assert "j < top_k ? inclusive[j] - weights[j] : total_top_k" in KERNEL
    assert "j < top_k ? inclusive[j] : total_top_k" in KERNEL
    assert "below < kept - 1 ? below : kept - 1" in KERNEL
    # the presence penalty from the row's history bits, the drawn token's bit set, the greedy tile copied
    assert "if ((hist[ids[j] >> 5] >> (ids[j] & 31)) & 1u)" in KERNEL
    assert "hist[chosen >> 5] |= 1u << (chosen & 31);" in KERNEL
    assert "if (greedy_flag) {" in KERNEL and KERNEL.count("noc.async_write(stage, token, TILE_BYTES") == 2
    # the working arrays live in L1, not on the RISC stack
    assert "volatile tt_l1_ptr uint32_t* values = words + WORK_VALUES;" in KERNEL
    assert "tile[(r >> 4) * 256 + (r & 15)] = as_bits(static_cast<float>(chosen));" in KERNEL


def test_policy_admits_presence_only_on_the_fused_sampler():
    instruct = Qwen38SamplingParameters.official_non_thinking(seed=3)
    thinking = Qwen38SamplingParameters.official_thinking(seed=3)
    assert ds.Qwen38DeviceSamplerPolicy.from_parameters(instruct) is None
    assert "presence penalty on a sampler without the device history" == ds.Qwen38DeviceSamplerPolicy.refusal(instruct)
    fused_policy = ds.Qwen38DeviceSamplerPolicy.from_parameters(instruct, presence_on_device=True)
    assert fused_policy == ds.Qwen38DeviceSamplerPolicy(
        temperature=0.7, top_k=20, top_p=0.8, min_p=0.0, presence_penalty=1.5
    )
    assert ds.Qwen38DeviceSamplerPolicy.from_parameters(thinking).presence_penalty == 0.0
    for parameters, reason in (
        (
            Qwen38SamplingParameters(
                temperature=1.0, top_p=0.9, top_k=20, presence_penalty=0.0, seed=1, frequency_penalty=0.5
            ),
            "frequency",
        ),
        (
            Qwen38SamplingParameters(
                temperature=1.0, top_p=0.9, top_k=20, presence_penalty=0.0, seed=1, repetition_penalty=1.2
            ),
            "repetition",
        ),
        (
            Qwen38SamplingParameters(temperature=1.0, top_p=0.9, top_k=20, presence_penalty=-0.5, seed=1),
            "negative presence",
        ),
        (Qwen38SamplingParameters(temperature=1.0, top_p=0.9, top_k=0, presence_penalty=0.0, seed=1), "top_k 0"),
        (
            Qwen38SamplingParameters(temperature=5.0, top_p=0.9, top_k=20, presence_penalty=0.0, seed=1),
            "temperature 5.0",
        ),
    ):
        assert ds.Qwen38DeviceSamplerPolicy.from_parameters(parameters, presence_on_device=True) is None
        assert reason in ds.Qwen38DeviceSamplerPolicy.refusal(parameters, presence_on_device=True)


def test_reference_takes_the_presence_penalty_off_the_seen_lanes_first():
    g = torch.Generator().manual_seed(2)
    values = (torch.randn(ds.LANES, generator=g) * 2 + 10).to(torch.bfloat16).float()
    ids = torch.randperm(VOCAB_SIZE, generator=g)[: ds.LANES]
    policy = ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=32, top_p=1.0, min_p=0.0, presence_penalty=1.5)
    top = int(torch.argmax(values))
    seen = torch.zeros(ds.LANES, dtype=torch.bool)
    seen[top] = True
    penalized = values.clone()
    penalized[top] -= 1.5
    plain = ds.device_sampler_reference(
        penalized, ids, ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=32, top_p=1.0, min_p=0.0), 0.0
    )
    with_history = ds.device_sampler_reference(values, ids, policy, 0.0, seen=seen)
    assert with_history.token_id == plain.token_id
    try:
        ds.device_sampler_reference(values, ids, policy, 0.0)
    except ValueError as error:
        assert "seen mask" in str(error)
    else:
        raise AssertionError("a presence policy without the seen mask must be refused")


def test_history_image_sets_one_bit_per_token_per_row():
    image = st.history_image(2, [[0, 31, 32, VOCAB_SIZE - 1], [5]])
    assert tuple(image.shape) == (1, 1, 2, st.HIST_WORDS) and image.dtype == torch.int32
    words = image.reshape(2, -1).to(torch.int64) & 0xFFFFFFFF
    assert int(words[0, 0]) == (1 | (1 << 31)) and int(words[0, 1]) == 1 and int(words[1, 0]) == 1 << 5
    assert int(words[0, (VOCAB_SIZE - 1) >> 5]) == 1 << ((VOCAB_SIZE - 1) & 31)
    assert int((words != 0).sum()) == 4
