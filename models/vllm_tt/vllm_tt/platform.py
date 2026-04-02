import logging
import os
from contextlib import contextmanager

import torch
from vllm.platforms.interface import Platform, PlatformEnum

logger = logging.getLogger(__name__)


def tt_platform_plugin() -> str | None:
    if os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "tt":
        return "vllm_tt.platform.TTPlatform"
    return None


class TTPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED

    device_name: str = "tt"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    supported_quantization: list[str] = []
    simple_compile_backend: str = "eager"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None = None) -> bool:
        return True

    @classmethod
    @contextmanager
    def inference_mode(cls):
        with torch.no_grad():
            yield

    @classmethod
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        from vllm_tt.models import register_tt_models

        parallel_config = vllm_config.parallel_config
        assert (
            parallel_config.tensor_parallel_size == 1
        ), "TT plugin does not support tensor parallelism"
        assert (
            parallel_config.pipeline_parallel_size == 1
        ), "TT plugin does not support pipeline parallelism"

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_tt.worker.TTWorker"

        register_tt_models()

        logger.info("TTPlatform configured: worker_cls=%s", parallel_config.worker_cls)

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        raise ValueError(f"TT plugin does not support quantization: {quant}")
