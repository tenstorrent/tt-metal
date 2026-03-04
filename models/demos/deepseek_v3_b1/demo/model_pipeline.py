from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.cli import CacheWeightProvider, SyntheticWeightProvider, open_mesh_device
from models.demos.deepseek_v3_b1.demo.pipeline import WeightProvider, create_pipeline_configuration_from_num_procs
from models.demos.deepseek_v3_b1.model import TOKEN_ID_BYTES, DeepSeekV3, page_size_bytes, to_padded_input


class ModelPipeline:
    def __init__(
        self, use_real_weights: bool, lm_head_fp32_dest_acc_en: bool, lm_head_persistent_mode: str, cache_path: Path
    ):
        logger.info(
            "Starting DeepSeek V3 B1 demo pod pipeline (weights={}, lm_head_fp32={}, lm_head_persistent_mode={})",
            "real" if use_real_weights else "synthetic",
            lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode,
        )
        if not is_slow_dispatch():
            raise RuntimeError(
                "DeepSeek V3 B1 demo requires slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
            )
        self.mesh_device = open_mesh_device()
        num_procs = int(ttnn.distributed_context_get_size())
        if num_procs not in (4, 16, 64):
            raise RuntimeError(f"Pod pipeline requires 4 or 16 distributed processes; got {num_procs}")
        ttnn.enable_asynchronous_slow_dispatch(self.mesh_device)

        self.provider: WeightProvider = (
            CacheWeightProvider(cache_path) if use_real_weights else SyntheticWeightProvider()
        )
        config = create_pipeline_configuration_from_num_procs(
            num_procs,
            self.provider,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            # dense_layer_id_override=dense_layer_id_override,
            # moe_layer_id_override=moe_layer_id_override,
        )
        assert (
            config.num_stages == num_procs
        ), f"Pipeline configuration has {config.num_stages} stages but {num_procs} processes"

        logger.info(f"Building pipeline")
        self.pipeline = config.build_pipeline(self.mesh_device)

        logger.info(f"Setting up and running pipeline")
        self.pipeline.setup_and_run()
        self.warmup()

    def warmup(self):
        if self.pipeline.my_mesh_id != 0:
            print("Warmup: skipping on non-zero mesh ID")
            return

        self.model = DeepSeekV3(
            write_fn=self.pipeline.write_token,
            read_fn=self.pipeline.read_output,
            batch_size=1,
        )

    def run_inference(self, prompt_ids, max_tokens):
        print("running inference")
        logger.debug(f"Prefilling...")
        if not prompt_ids:
            raise ValueError("No token ids provided")
        page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        prompt_token_tensors = [
            to_padded_input(
                torch.tensor([[tid]], dtype=torch.int32),
                batch_size=1,
                page_size_datums=page_size_datums,
            )
            for tid in prompt_ids
        ]
        last_output = self.model.prefill(prompt_token_tensors)
        next_token_id = int(ttnn.to_torch(last_output).to(torch.int32)[0, 0].item())
        generated = [next_token_id]
        yield next_token_id

        for step in range(max_tokens - 1):
            output = self.model.decode_step(
                torch.tensor([[next_token_id]], dtype=torch.int32),
            )
            next_token_id = int(ttnn.to_torch(output).to(torch.int32)[0, 0].item())
            generated.append(next_token_id)
            logger.info("Decode step {} output token: {}", step + 1, next_token_id)
            yield next_token_id

        logger.info("Generated {} tokens total", len(generated))
