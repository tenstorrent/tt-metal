import ttnn
import tests.ttnn.unit_tests.operations.greater_than.utils as utils
import torch
import torch.nn.functional as F
from loguru import logger

DRAM_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float64) if x.dtype != torch.float64 else x
    y_float = y.to(torch.float64) if y.dtype != torch.float64 else y

    x_flat, y_flat = x_float.flatten(), y_float.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")
    else:
        return ((vx @ vy) / denom).item()


class Cpu_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score_thresh = 0.01

    def forward(self, cls_logits):
        pred_scores = F.softmax(cls_logits, dim=-1)
        for scores in pred_scores:
            for label in range(1, 2):
                score = scores[:, label]
                keep_idxs = score > self.score_thresh
        return pred_scores, score, keep_idxs


def test_gt_org():
    device = utils.DeviceGetter.get_device((1, 1))

    # Load input tensor
    logger.info("Loading input tensor...")
    input_ttnn = utils.load_tensor("arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.FLOAT32, device, DRAM_MEMCFG)
    input_torch = ttnn.to_torch(input_ttnn).to(torch.float32)

    # === TT inference ===
    logger.info("Running TT inference...")

    # Softmax
    tt_tiled = ttnn.to_layout(input_ttnn, ttnn.Layout.TILE, None, memory_config=DRAM_MEMCFG)
    tt_softmax = ttnn.softmax(tt_tiled, 2, memory_config=DRAM_MEMCFG)
    tt_softmax_torch = ttnn.to_torch(tt_softmax)

    # Slice column 1: [0,0,1] to [1,8732,2]
    tt_slice = ttnn.slice(tt_softmax, [0, 0, 1], [1, 8732, 2], [1, 1, 1], memory_config=DRAM_MEMCFG)
    tt_slice_torch = ttnn.to_torch(tt_slice)

    # Reshape to [8732]
    tt_reshape = ttnn.reshape(tt_slice, [8732], memory_config=DRAM_MEMCFG)
    tt_reshape_torch = ttnn.to_torch(tt_reshape)

    # Greater than 0.01
    threshold = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=0.0099999997764825821,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMCFG,
    )
    tt_gt = ttnn.gt(tt_reshape, threshold, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMCFG)
    tt_gt_torch = ttnn.to_torch(tt_gt)

    # === CPU golden inference ===
    logger.info("Running CPU inference...")
    cpu_model = Cpu_model()
    cpu_model.eval()

    with torch.no_grad():
        cpu_softmax, cpu_score, cpu_gt = cpu_model(input_torch)

    # === Intermediate: Softmax ===
    logger.info("=== Intermediate Output Comparison (Softmax) ===")
    softmax_pcc = compute_pcc(tt_softmax_torch, cpu_softmax)
    logger.info(f"Softmax Output PCC: {softmax_pcc}")

    # === Intermediate: Slice+Reshape (TT) == scores[:,label] (CPU) ===
    logger.info("=== Intermediate Output Comparison (Slice+Reshape) ===")
    slice_reshape_pcc = compute_pcc(tt_reshape_torch, cpu_score)
    logger.info(f"Slice+Reshape Output PCC: {slice_reshape_pcc}")

    # === Final: Greater Than ===
    logger.info("=== Final Output Comparison (Greater Than) ===")
    gt_pcc = compute_pcc(tt_gt_torch, cpu_gt)
    logger.info(f"GT Output PCC: {gt_pcc}")
