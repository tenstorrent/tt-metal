import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize.tilize import tilize
from models.common.utility_functions import comp_pcc

dev = ttnn.open_device(device_id=0)
torch.manual_seed(42)
base_u, base_d = pd.UNPACK_TO_DEST_DTYPES, pd.DEST32_DTYPES
for di, mk in [
    (ttnn.bfloat16, lambda s: torch.randn(s).bfloat16()),
    (ttnn.float32, lambda s: torch.randn(s)),
    (ttnn.fp8_e4m3, lambda s: torch.randn(s).to(torch.float8_e4m3fn).float()),
]:
    for pv in [-7.0, 3.0, 0.0]:
        torch.manual_seed(42)
        x = mk((1, 1, 50, 50))
        exp = torch.nn.functional.pad(x.float(), (0, 14, 0, 14), value=pv)
        for name, (u, d) in {"cur": (base_u, base_d), "dest16": (tuple(), (ttnn.fp8_e4m3,))}.items():
            if di == ttnn.fp8_e4m3 and name == "dest16":
                continue
            pd.UNPACK_TO_DEST_DTYPES, pd.DEST32_DTYPES = u, d
            try:
                t = ttnn.from_torch(x, dtype=di, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
                y = tilize(t, dtype=ttnn.bfloat4_b, pad_value=pv).cpu().to_torch_with_padded_shape().float()
                # host reference: ttnn's own host bfp4 conversion of the padded tensor
                h = ttnn.to_torch(ttnn.from_torch(exp, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
                print(
                    "RES",
                    di,
                    pv,
                    name,
                    "pcc_dev",
                    round(comp_pcc(exp, y, 0.98)[1] if isinstance(comp_pcc(exp, y, 0.98)[1], float) else 0, 5),
                    comp_pcc(exp, y, 0.98)[1],
                    "pcc_host",
                    comp_pcc(exp, h, 0.98)[1],
                    "dev==host",
                    torch.equal(y, h),
                )
            except Exception as e:
                print("RES err", di, pv, name, str(e)[:200])
pd.UNPACK_TO_DEST_DTYPES, pd.DEST32_DTYPES = base_u, base_d
ttnn.close_device(dev)
