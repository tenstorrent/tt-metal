# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.llama.tt.llama_first_half import (
    TtLlamaModelFirstHFModel,
)
from models.experimental.llama.tt.llama_second_half import (
    TtLlamaModelSecondHFModel,
)


def _llama_first_half(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
):
    return TtLlamaModelFirstHFModel(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start,
        num_decoders,
    )


def llama_first_half(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
) -> TtLlamaModelFirstHFModel:
    return _llama_first_half(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start,
        num_decoders,
    )


def _llama_second_half(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
    is_causallm=False,
):
    return TtLlamaModelSecondHFModel(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start,
        num_decoders,
        is_causallm,
    )


def llama_second_half(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
    is_causallm=True,
) -> TtLlamaModelSecondHFModel:
    return _llama_second_half(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start,
        num_decoders,
        is_causallm,
    )
