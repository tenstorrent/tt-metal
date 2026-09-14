# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Speed of FIBO-vlm: sampled decode, and the prefill of an image prompt through tower and encoder.

Timing only: the weights are random, the shapes are the checkpoint's.
"""

from __future__ import annotations

import pathlib
import time

import pytest
import torch
import transformers
from loguru import logger
from PIL import Image

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.tt_dit.encoders.qwen3vl.model_qwen3vl import Qwen3VlEncoder, mrope_position_ids
from models.tt_dit.encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, vision_cu_seqlens
from models.tt_dit.encoders.transformer import MAX_CHUNK_SIZE, Cache
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor

CHECKPOINT = "briaai/FIBO-vlm"
PROMPT_LENGTH = 1024
NUM_STEPS = 32
# The image and prompt of the tt_transformers demo's `sample_prompts/demo_1.json`, so that both sides
# see the same patch grid.
IMAGE = pathlib.Path(__file__).parents[5] / "models/sample_data/house_in_field_1080p.jpg"
IMAGE_PROMPT = "Describe this image."

MESH = pytest.mark.parametrize(
    "mesh_device", [pytest.param((1, 4), id="1x4"), pytest.param((1, 8), id="1x8")], indirect=True
)
DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 32_000_000}],
    indirect=True,
)


def _encoder(hf_config: transformers.PretrainedConfig, mesh_device: ttnn.MeshDevice) -> Qwen3VlEncoder:
    """The text model with random weights, tensor-parallel over the mesh."""
    tp_axis = 1
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=default_topology(mesh_device) or ttnn.Topology.Linear)
    encoder = Qwen3VlEncoder(
        Qwen3VlEncoder.config_from_hf(hf_config),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    logger.info("building random torch model...")
    torch_model = transformers.AutoModel.from_config(hf_config.text_config, dtype=torch.bfloat16)
    state = {f"model.language_model.{k}": v for k, v in torch_model.state_dict().items()}
    # The checkpoint ties the lm head to the token embedding.
    state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    encoder.load_torch_state_dict(Qwen3VlEncoder.convert_state(state))
    return encoder


def _tower(vision_config: transformers.PretrainedConfig, mesh_device: ttnn.MeshDevice) -> Qwen3VlVisionModel:
    """The vision tower with random weights, replicated over the mesh like the tt_transformers one."""
    tower = Qwen3VlVisionModel(
        hidden_size=vision_config.hidden_size,
        num_heads=vision_config.num_heads,
        depth=vision_config.depth,
        intermediate_size=vision_config.intermediate_size,
        in_channels=vision_config.in_channels,
        patch_size=vision_config.patch_size,
        temporal_patch_size=vision_config.temporal_patch_size,
        spatial_merge_size=vision_config.spatial_merge_size,
        num_position_embeddings=vision_config.num_position_embeddings,
        out_hidden_size=vision_config.out_hidden_size,
        hidden_act=vision_config.hidden_act,
        deepstack_visual_indexes=vision_config.deepstack_visual_indexes,
        mesh_device=mesh_device,
    )
    torch_model = transformers.AutoModel.from_config(vision_config, dtype=torch.bfloat16)
    tower.load_torch_state_dict(torch_model.state_dict())
    return tower


@MESH
@DEVICE_PARAMS
def test_decode(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    max_length = PROMPT_LENGTH + NUM_STEPS

    hf_config = transformers.AutoConfig.from_pretrained(CHECKPOINT)
    encoder = _encoder(hf_config, mesh_device)

    torch_prompt = torch.randint(0, hf_config.text_config.vocab_size, (1, PROMPT_LENGTH))
    prompt = tensor.from_torch(torch_prompt, device=mesh_device, dtype=ttnn.uint32)

    # As in `generate`: a fresh cache sized for the whole sequence, the prompt prefilled into it.
    padded_length = -(-max_length // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE

    prefill_times = []
    for _ in range(2):
        start = time.perf_counter()
        encoder.forward(
            prompt,
            cache=Cache(device=mesh_device, size=padded_length, batch_size=1),
            skip_final_linear=True,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_times.append(time.perf_counter() - start)

    logger.info(
        f"prefill {PROMPT_LENGTH} tokens: {prefill_times[-1] * 1e3:.1f} ms (first {prefill_times[0] * 1e3:.1f} ms)"
    )
    _time_generation(encoder, torch_prompt, prefill_time=prefill_times[-1])


def _time_generation(encoder: Qwen3VlEncoder, tokens: torch.Tensor, *, prefill_time: float, **kwargs: object) -> None:
    """Logs `generate` of NUM_STEPS tokens after `tokens`, untraced and traced, with `kwargs` passed on."""
    generation_config = transformers.GenerationConfig.from_pretrained(CHECKPOINT)
    generated = {}
    for traced in (False, True):
        # Traced twice: the first call captures the trace, the second replays it.
        totals = []
        for _ in range(2 if traced else 1):
            torch.manual_seed(0)
            start = time.perf_counter()
            generated[traced] = encoder.generate(
                tokens,
                mask=None,
                max_length=tokens.shape[1] + NUM_STEPS,
                eos_tokens=None,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                temperature=generation_config.temperature,
                traced=traced,
                **kwargs,
            ).tokens
            totals.append(time.perf_counter() - start)
        mode = "traced" if traced else "untraced"
        for total, run in zip(totals, ("first", "second"), strict=False):
            logger.info(
                f"generate, {NUM_STEPS} tokens end to end, {mode}, {run}: {total * 1e3:.1f} ms, "
                f"{(total - prefill_time) / NUM_STEPS * 1e3:.2f} ms per token after the prefill"
            )
    mismatches = generated[True].ne(generated[False]).sum().item()
    logger.info(f"sampled tokens with the same seed, traced against untraced: {mismatches} of {NUM_STEPS} differ")


@MESH
@DEVICE_PARAMS
def test_image_prompt(*, mesh_device: ttnn.MeshDevice) -> None:
    """Times the tower, the text prefill and the generation after the image.

    The tower and the prefill are timed separately, as the tt_transformers demo reports them. The
    demo prefill also runs the lm head on the last token, which the timed encoder prefill skips.
    """
    torch.set_num_threads(1)
    torch.manual_seed(0)

    hf_config = transformers.AutoConfig.from_pretrained(CHECKPOINT)
    processor = transformers.AutoProcessor.from_pretrained(CHECKPOINT)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": IMAGE_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[Image.open(IMAGE)], return_tensors="pt")
    ids, pixel_values, grid = inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw
    vision_mask = ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    position_ids = mrope_position_ids(
        vision_mask.long(), image_grid_thw=grid, spatial_merge_size=hf_config.vision_config.spatial_merge_size
    )
    logger.info(
        f"prompt of {ids.shape[1]} tokens, {int(vision_mask.sum())} of them image tokens, grid {grid[0].tolist()}"
    )

    tower = _tower(hf_config.vision_config, mesh_device)
    encoder = _encoder(hf_config, mesh_device)

    for run in ("first", "second"):
        # The host-side preparation counts towards the tower, as it does in the demo.
        start = time.perf_counter()
        cos, sin = tower.prepare_rope(grid)
        vision_embeds, deepstack_embeds = tower.forward(
            tensor.from_torch(pixel_values, device=mesh_device),
            pos_embeds=tensor.from_torch(tower.prepare_pos_embeds(grid), device=mesh_device),
            rope=(tensor.from_torch(cos, device=mesh_device), tensor.from_torch(sin, device=mesh_device)),
            cu_seqlens=vision_cu_seqlens(grid),
        )
        ttnn.synchronize_device(mesh_device)
        tower_time = time.perf_counter() - start

        start = time.perf_counter()
        encoder.forward(
            tensor.from_torch(ids, device=mesh_device, dtype=ttnn.uint32),
            positions=tensor.from_torch(position_ids.float(), device=mesh_device, dtype=ttnn.float32),
            vision_embeds=vision_embeds,
            vision_mask=tensor.from_torch(vision_mask, device=mesh_device),
            deepstack_embeds=deepstack_embeds,
            skip_final_linear=True,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_time = time.perf_counter() - start

        logger.info(
            f"{run}: tower {tower_time * 1e3:.1f} ms, prefill {prefill_time * 1e3:.1f} ms, "
            f"total {(tower_time + prefill_time) * 1e3:.1f} ms"
        )

    _time_generation(
        encoder,
        ids,
        prefill_time=prefill_time,
        positions=position_ids,
        vision_embeds=vision_embeds,
        vision_mask=vision_mask,
        deepstack_embeds=deepstack_embeds,
    )
