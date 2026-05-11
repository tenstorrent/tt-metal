from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PatchifyMetadata:
    original_seq_len: int
    pad_length: int
    patch_size: int


def _torch_rmsnorm_qwen3(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Match transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm (as used in repo tests)
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * weight.float()).to(x.dtype)


class TorchProjOut(nn.Module):
    def __init__(self, *, patch_size: int, convt: nn.ConvTranspose1d):
        super().__init__()
        self.patch_size = int(patch_size)
        self.convt = convt

    def forward(self, x: torch.Tensor, *, original_seq_len: int) -> torch.Tensor:
        # x: [B, T_p, hidden]
        x = x.transpose(1, 2)  # [B, hidden, T_p]
        x = self.convt(x)  # [B, out_ch, T]
        x = x.transpose(1, 2)  # [B, T, out_ch]
        return x[:, :original_seq_len, :]


class TorchPatchEmbed1D(nn.Module):
    """
    Torch equivalent of HF `proj_in` patch embedding in AceStepDiTModel.

    Input:  [B, T, in_channels]
    Output: [B, T_p, hidden_size]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.dtype = dtype

        conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        ).to(dtype)
        with torch.no_grad():
            conv.weight.copy_(weight.to(dtype))
            conv.bias.copy_(bias.to(dtype))
        self.proj_in = conv.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PatchifyMetadata]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,T,C], got {tuple(x.shape)}")
        B, T, C = x.shape
        if int(C) != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got C={C}")

        remainder = int(T) % self.patch_size
        pad_length = 0 if remainder == 0 else (self.patch_size - remainder)
        if pad_length:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_length), mode="constant", value=0.0)
        meta = PatchifyMetadata(original_seq_len=int(T), pad_length=int(pad_length), patch_size=self.patch_size)

        x = x.transpose(1, 2)  # [B,C,T]
        x = self.proj_in(x)  # [B,hidden,T_p]
        x = x.transpose(1, 2)  # [B,T_p,hidden]
        return x, meta


class TorchAceStepDiTOutputHead(nn.Module):
    """
    Minimal Torch reference for the HF AceStepDiTModel tail:
      norm_out (RMSNorm) -> timestep-conditioned scale/shift -> proj_out (ConvTranspose1d) -> crop
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        out_channels: int,
        patch_size: int,
        eps: float,
        norm_weight: torch.Tensor,
        scale_shift_table: torch.Tensor,
        proj_out_weight: torch.Tensor,
        proj_out_bias: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.eps = float(eps)
        self.dtype = dtype

        if tuple(scale_shift_table.shape) != (1, 2, self.hidden_size):
            raise ValueError(
                f"scale_shift_table must be [1,2,{self.hidden_size}], got {tuple(scale_shift_table.shape)}"
            )
        if int(norm_weight.shape[0]) != self.hidden_size:
            raise ValueError(f"norm_weight must be [{self.hidden_size}], got {tuple(norm_weight.shape)}")

        self.register_buffer("norm_weight", norm_weight.to(dtype), persistent=False)
        self.register_buffer("scale_shift_table", scale_shift_table.to(dtype), persistent=False)

        convt = nn.ConvTranspose1d(
            in_channels=self.hidden_size,
            out_channels=self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        ).to(dtype)
        with torch.no_grad():
            convt.weight.copy_(proj_out_weight.to(dtype))
            convt.bias.copy_(proj_out_bias.to(dtype))
        self.proj_out = TorchProjOut(patch_size=self.patch_size, convt=convt).eval()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, meta: PatchifyMetadata) -> torch.Tensor:
        # x: [B, T_p, hidden_size]
        # temb: [B, hidden_size]
        normed = _torch_rmsnorm_qwen3(x, self.norm_weight, self.eps)
        shift = self.scale_shift_table[:, 0:1, :] + temb.unsqueeze(1)
        scale = self.scale_shift_table[:, 1:2, :] + temb.unsqueeze(1)
        modulated = (normed * (1 + scale) + shift).type_as(x)
        return self.proj_out(modulated, original_seq_len=meta.original_seq_len)


def _read_hf_config(snapshot_dir: Path) -> dict:
    cfg_path = snapshot_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected config.json in snapshot: {cfg_path}")
    return json.loads(cfg_path.read_text())


def _pick_output_head_prefix(keys: list[str]) -> str:
    suffix = ".norm_out.weight"
    candidates: list[str] = []
    for k in keys:
        if k.endswith(suffix):
            candidates.append(k[: -len(suffix)])
    # Prefer a prefix that also has scale_shift_table and proj_out weights
    for base in candidates:
        if f"{base}.scale_shift_table" in keys:
            if f"{base}.proj_out.1.weight" in keys and f"{base}.proj_out.1.bias" in keys:
                return base
            if f"{base}.proj_out.weight" in keys and f"{base}.proj_out.bias" in keys:
                return base
    if candidates:
        return candidates[0]
    raise KeyError("Could not find any *.norm_out.weight key to infer output-head prefix.")


def _load_required_tensors_from_safetensors(snapshot_dir: Path) -> tuple[dict, str]:
    try:
        from safetensors import safe_open
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This demo requires safetensors. Install it in your env.") from e

    # ACE-Step checkpoints can be nested under a subfolder (e.g. acestep-v15-turbo/model.safetensors)
    st_files = sorted(snapshot_dir.rglob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found under snapshot dir: {snapshot_dir}")

    # Pick the first shard and use its key list to infer prefix; then read tensors from any shard where they exist.
    all_keys: set[str] = set()
    for p in st_files:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            all_keys.update(list(f.keys()))
    keys_list = sorted(all_keys)
    base = _pick_output_head_prefix(keys_list)

    # Candidate key patterns (HF sometimes uses sequential wrappers)
    k_norm = f"{base}.norm_out.weight"
    k_sst = f"{base}.scale_shift_table"
    k_pin_w1 = f"{base}.proj_in.1.weight"
    k_pin_b1 = f"{base}.proj_in.1.bias"
    k_pin_w0 = f"{base}.proj_in.weight"
    k_pin_b0 = f"{base}.proj_in.bias"
    k_w1 = f"{base}.proj_out.1.weight"
    k_b1 = f"{base}.proj_out.1.bias"
    k_w0 = f"{base}.proj_out.weight"
    k_b0 = f"{base}.proj_out.bias"

    want = [k_norm, k_sst]
    if k_pin_w1 in all_keys and k_pin_b1 in all_keys:
        want += [k_pin_w1, k_pin_b1]
        pin_wk, pin_bk = k_pin_w1, k_pin_b1
    elif k_pin_w0 in all_keys and k_pin_b0 in all_keys:
        want += [k_pin_w0, k_pin_b0]
        pin_wk, pin_bk = k_pin_w0, k_pin_b0
    else:
        raise KeyError(f"Could not find proj_in weights under prefix '{base}'.")

    if k_w1 in all_keys and k_b1 in all_keys:
        want += [k_w1, k_b1]
        proj_wk, proj_bk = k_w1, k_b1
    elif k_w0 in all_keys and k_b0 in all_keys:
        want += [k_w0, k_b0]
        proj_wk, proj_bk = k_w0, k_b0
    else:
        raise KeyError(f"Could not find proj_out weights under prefix '{base}'.")

    loaded: dict[str, torch.Tensor] = {}
    remaining = set(want)
    for p in st_files:
        if not remaining:
            break
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in list(remaining):
                if k in f.keys():
                    loaded[k] = f.get_tensor(k)
                    remaining.remove(k)
    if remaining:
        raise KeyError(f"Missing required keys from safetensors shards: {sorted(remaining)}")

    return (
        {
            "base_prefix": base,
            "norm_weight": loaded[k_norm],
            "scale_shift_table": loaded[k_sst],
            "proj_in_weight": loaded[pin_wk],
            "proj_in_bias": loaded[pin_bk],
            "proj_out_weight": loaded[proj_wk],
            "proj_out_bias": loaded[proj_bk],
        },
        base,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="ACE-Step v1.5 Torch demo using HF-downloaded weights.")
    ap.add_argument("--repo-id", default="ACE-Step/Ace-Step1.5", help="Hugging Face repo id to download from.")
    ap.add_argument(
        "--subfolder",
        default="acestep-v15-turbo",
        help="Optional subfolder inside snapshot to load (ACE-Step repos often contain multiple variants).",
    )
    ap.add_argument("--revision", default=None, help="Optional HF revision (branch/tag/commit).")
    ap.add_argument("--cache-dir", default=None, help="Optional HF cache dir override.")
    ap.add_argument("--seed", type=int, default=0, help="Torch RNG seed for deterministic demo input.")
    ap.add_argument(
        "--input-pt",
        default=None,
        help="Optional path to a torch-saved Tensor (.pt) containing input latents (expected [T,C] or [B,T,C]).",
    )
    ap.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Stddev for Gaussian noise. If --input-pt is provided, noise is added on top.",
    )
    ap.add_argument(
        "--original-seq-len",
        type=int,
        default=None,
        help="Original (unpadded) sequence length. If omitted and --input-pt is provided, inferred from input.",
    )
    ap.add_argument("--batch", type=int, default=1, help="Batch size for demo input.")
    args = ap.parse_args()

    try:
        pass
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This demo requires huggingface_hub. Install it in your env.") from e

    sig = run_hf_output_head_demo(
        repo_id=args.repo_id,
        subfolder=args.subfolder,
        revision=args.revision,
        cache_dir=args.cache_dir,
        seed=args.seed,
        batch=args.batch,
        original_seq_len=args.original_seq_len,
        input_pt=args.input_pt,
        noise_std=args.noise_std,
        offline=bool(os.environ.get("HF_HUB_OFFLINE")),
    )

    print(json.dumps(sig, indent=2, sort_keys=True))
    return 0


def run_hf_output_head_demo(
    *,
    repo_id: str,
    subfolder: str | None = "acestep-v15-turbo",
    revision: str | None = None,
    cache_dir: str | None = None,
    seed: int = 0,
    batch: int = 1,
    original_seq_len: int | None = None,
    input_pt: str | None = None,
    noise_std: float = 1.0,
    offline: bool = False,
) -> dict:
    """
    Programmatic entrypoint used by pytest.
    Returns the same JSON-serializable signature dict printed by `main()`.
    """
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
    )

    model_dir = snapshot_dir / subfolder if subfolder else snapshot_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Requested subfolder does not exist in snapshot: {model_dir}")

    hf_cfg = _read_hf_config(model_dir)
    patch_size = int(hf_cfg.get("patch_size", 2))
    hidden_size = int(hf_cfg.get("hidden_size", hf_cfg.get("dim", 2048)))
    out_channels = int(hf_cfg.get("audio_acoustic_hidden_dim", hf_cfg.get("out_channels", 64)))
    eps = float(hf_cfg.get("rms_norm_eps", 1e-6))

    tensors, base = _load_required_tensors_from_safetensors(model_dir)

    torch.manual_seed(int(seed))
    B = int(batch)

    # Build input in latent space [B,T,in_channels] (noise by default), then patchify -> output head.
    if input_pt is not None:
        obj = torch.load(input_pt, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            x0 = obj
        elif isinstance(obj, dict):
            vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if len(vals) != 1:
                raise TypeError(f"--input-pt dict payload must contain exactly 1 tensor, got keys={list(obj.keys())}")
            x0 = vals[0]
        else:
            raise TypeError(f"Unsupported --input-pt payload type: {type(obj)}")

        if x0.ndim == 2:  # [T,C]
            x0 = x0.unsqueeze(0)  # [1,T,C]
        if x0.ndim != 3:
            raise ValueError(f"--input-pt tensor must be [T,C] or [B,T,C], got {tuple(x0.shape)}")
        if B != int(x0.shape[0]):
            if int(x0.shape[0]) == 1 and B > 1:
                x0 = x0.expand(B, -1, -1).contiguous()
            else:
                raise ValueError(f"--batch={B} does not match input batch={int(x0.shape[0])}")
        x = x0.to(torch.bfloat16)
        inferred_T = int(x.shape[1])
        orig_T = int(original_seq_len) if original_seq_len is not None else inferred_T
        if orig_T != inferred_T:
            raise ValueError(f"--original-seq-len={orig_T} must match input T={inferred_T} for this demo")
        original_seq_len = orig_T
    else:
        original_seq_len = int(original_seq_len) if original_seq_len is not None else 257
        x = torch.randn((B, original_seq_len, int(hf_cfg.get("in_channels", 192))), dtype=torch.bfloat16)

    if float(noise_std) != 0.0:
        x = x + torch.randn_like(x) * float(noise_std)

    in_channels = int(hf_cfg.get("in_channels", 192))
    if int(x.shape[-1]) != in_channels:
        raise ValueError(f"Input channels mismatch: got C={int(x.shape[-1])}, expected in_channels={in_channels}")

    patch_embed = TorchPatchEmbed1D(
        in_channels=in_channels,
        hidden_size=hidden_size,
        patch_size=patch_size,
        weight=tensors["proj_in_weight"],
        bias=tensors["proj_in_bias"],
        dtype=torch.bfloat16,
    ).eval()

    patches, meta = patch_embed(x)
    temb = torch.randn((B, hidden_size), dtype=torch.bfloat16)

    head = TorchAceStepDiTOutputHead(
        hidden_size=hidden_size,
        out_channels=out_channels,
        patch_size=patch_size,
        eps=eps,
        norm_weight=tensors["norm_weight"],
        scale_shift_table=tensors["scale_shift_table"],
        proj_out_weight=tensors["proj_out_weight"],
        proj_out_bias=tensors["proj_out_bias"],
        dtype=torch.bfloat16,
    ).eval()

    with torch.no_grad():
        y = head(patches, temb, meta)

    y_f = y.float()
    t_p = int(patches.shape[1])
    return {
        "repo_id": repo_id,
        "snapshot_dir": str(snapshot_dir),
        "model_dir": str(model_dir),
        "picked_prefix": base,
        "config": {
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "eps": eps,
        },
        "input": {
            "input_pt": input_pt,
            "noise_std": float(noise_std),
            "B": B,
            "T": int(x.shape[1]),
            "C": int(x.shape[2]),
            "T_p": t_p,
            "original_seq_len": int(original_seq_len),
        },
        "output": {
            "shape": list(y.shape),
            "mean": float(y_f.mean()),
            "std": float(y_f.std(unbiased=False)),
            "first8": [float(v) for v in y_f.reshape(-1)[:8].tolist()],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
