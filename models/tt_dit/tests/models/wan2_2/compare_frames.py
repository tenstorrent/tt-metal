import sys

import numpy as np


def compare(ref_path, cand_path):
    ref = np.load(ref_path)
    cand = np.load(cand_path)

    print(f"Reference: {ref_path}  shape={ref.shape}  dtype={ref.dtype}")
    print(f"Candidate: {cand_path}  shape={cand.shape}  dtype={cand.dtype}")
    print(f"Ref range:  [{ref.min():.6f}, {ref.max():.6f}]")
    print(f"Cand range: [{cand.min():.6f}, {cand.max():.6f}]")
    print()

    r = ref.astype(np.float64).flatten()
    c = cand.astype(np.float64).flatten()
    diff = r - c

    pcc = np.corrcoef(r, c)[0, 1]
    mse = np.mean(diff**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    abs_err = np.abs(diff)
    max_abs = abs_err.max()
    mean_abs = abs_err.mean()
    p99_abs = np.percentile(abs_err, 99)

    print(f"PCC:            {pcc:.6f}")
    print(f"PSNR:           {psnr:.2f} dB")
    print(f"MSE:            {mse:.8f}")
    print(f"Max abs error:  {max_abs:.6f}")
    print(f"Mean abs error: {mean_abs:.6f}")
    print(f"P99 abs error:  {p99_abs:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <reference.npy> <candidate.npy>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
