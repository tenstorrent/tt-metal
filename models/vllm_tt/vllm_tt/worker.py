import logging
import random

import torch
from vllm.lora.request import LoRARequest
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

logger = logging.getLogger(__name__)


class TTWorker(WorkerBase):
    def init_device(self) -> None:
        self.device = torch.device("cpu")

    def load_model(self) -> None:
        from vllm.model_executor.model_loader import get_model

        self.model = get_model(vllm_config=self.vllm_config)
        logger.info("Loaded model: %s", type(self.model).__name__)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_kv_cache_spec(self) -> dict[str, FullAttentionSpec]:
        # Return a single-layer dummy spec so vLLM can compute block counts.
        # The no-op model doesn't actually use KV cache.
        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        return {
            "layer_0": FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.float16,
            )
        }

    def determine_available_memory(self) -> int:
        # Return a large value so vLLM allocates plenty of KV cache blocks.
        # No real memory profiling needed for the no-op model.
        return 4 * (1024**3)  # 4 GiB

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # No real KV cache to allocate for the no-op model.
        pass

    def compile_or_warm_up_model(self) -> float:
        return 0.0

    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None:
        # Two-phase execution: when tokens are scheduled, return None here
        # and let sample_tokens return the result. When nothing is scheduled,
        # return a ModelRunnerOutput directly (the engine won't call
        # sample_tokens in that case).
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())

        if not req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
            )

        req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        vocab_size = self.model_config.get_vocab_size()

        sampled_token_ids = []
        for _ in req_ids:
            token = random.randint(0, vocab_size - 1)
            sampled_token_ids.append([token])

        self._pending_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
        )
        return None

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        output = self._pending_output
        self._pending_output = None
        return output

    def check_health(self) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        spec = list(self.get_kv_cache_spec().values())[0]
        return spec.page_size_bytes

    def get_supported_tasks(self):
        return ("generate",)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError
