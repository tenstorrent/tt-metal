import torch
from loguru import logger


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float32) if x.dtype != torch.float32 else x
    y_float = y.to(torch.float32) if y.dtype != torch.float32 else y

    x_flat, y_flat = x_float.flatten(), y_float.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")
    else:
        return ((vx @ vy) / denom).item()


def test_comp():
    logger.info("=== Loading tensors ===")

    tt_xla_output = torch.load("tt_xla_output.pt", map_location="cpu")
    tt_xla_cpu_output = torch.load("tt_xla_cpu_output.pt", map_location="cpu")
    tt_metal_output = torch.load("tt_metal_output.pt", map_location="cpu")
    tt_metal_cpu_output = torch.load("tt_metal_cpu_output.pt", map_location="cpu")

    logger.info("✓ All tensors loaded")

    # ===== Comparison 1: TT-XLA output vs TT-Metal output =====

    logger.info("torch.allclose(tt_xla_output, tt_metal_output={}", torch.allclose(tt_xla_output, tt_metal_output))
    logger.info("pcc={}", compute_pcc(tt_xla_output, tt_metal_output))

    # ===== Comparison 2: TT-XLA cpu output vs TT-Metal cpu output =====

    logger.info(
        "torch.allclose(tt_xla_cpu_output, tt_metal_cpu_output={}",
        torch.allclose(tt_xla_cpu_output, tt_metal_cpu_output),
    )
    logger.info("pcc={}", compute_pcc(tt_xla_cpu_output, tt_metal_cpu_output))

    logger.info("tt_xla_output.shape={}", tt_xla_output.shape)
    logger.info("tt_metal_output.shape={}", tt_metal_output.shape)
    logger.info("tt_xla_cpu_output.shape={}", tt_xla_cpu_output.shape)
    logger.info("tt_metal_cpu_output.shape={}", tt_metal_cpu_output.shape)

    logger.info("tt_xla_output.dtype={}", tt_xla_output.dtype)
    logger.info("tt_metal_output.dtype={}", tt_metal_output.dtype)
    logger.info("tt_xla_cpu_output.dtype={}", tt_xla_cpu_output.dtype)
    logger.info("tt_metal_cpu_output.dtype={}", tt_metal_cpu_output.dtype)

    logger.info("tt_xla_output={}", tt_xla_output)
    logger.info("tt_metal_output={}", tt_metal_output)
    logger.info("tt_xla_cpu_output={}", tt_xla_cpu_output)
    logger.info("tt_metal_cpu_output={}", tt_metal_cpu_output)
