"""Diffusion-image correctness (Stable Diffusion / SDXL / FLUX).

Compares the demo's generated image against the HF CPU reference
using perceptual metrics:

* SSIM (structural similarity) — primary metric. Captures
  high-level structural agreement and is far more discriminating
  than per-pixel MSE for generative images. Pure-numpy (no deps).
* LPIPS (learned perceptual) — secondary, optional. Requires the
  ``lpips`` package; degraded gracefully to "skipped" if absent.

Why both
--------
SSIM alone is fooled by texture-matched but structurally-different
images (it can rate two random noise images close to 0.4); LPIPS
alone is too expensive to run on every iteration. Running SSIM
as the primary gate and LPIPS only when SSIM is borderline gives
us decent signal at low cost.

Demo-output protocol
--------------------
1. ``==IMAGE 0 - OUTPUT`` marker followed by either:
   * one line of base64-encoded PNG bytes, or
   * an ``image: <path>.png`` reference to an on-disk file.
2. Falls back to soft skip.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .base import Comparator, Evidence, ValidationResult
from .registry import register_comparator


DEFAULT_SSIM_MIN = 0.55
DEFAULT_LPIPS_MAX = 0.45


_IMAGE_MARKER_RE = re.compile(r"^==IMAGE\s+(?P<idx>\d+)\s+-\s+OUTPUT\s*$", re.M)
_IMAGE_PATH_RE = re.compile(r"^\s*image:\s*(?P<path>\S+\.(?:png|jpg|jpeg|webp))\s*$", re.M)


def _to_grayscale_np(img: Any) -> Any:
    """PIL Image or HxWx3 ndarray -> HxW float64 grayscale 0..1.
    Used internally by SSIM."""
    import numpy as np
    from PIL import Image

    if hasattr(img, "convert"):
        arr = np.array(img.convert("L"), dtype=np.float64) / 255.0
    else:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        if arr.max() > 1.0:
            arr = arr / 255.0
    return arr


def ssim(a: Any, b: Any) -> float:
    """Compute the (global) Structural Similarity Index between
    two images. Returns 1.0 for identical, ~0 for unrelated.

    Pure numpy. The implementation follows the original
    Wang et al. (2004) formulation without sliding window — a
    single-window global SSIM that's fast and good enough as a
    diffusion-comparison gate. For pixel-perfect agreement (TT
    matches HF byte-for-byte), this returns 1.0.
    """
    import numpy as np

    A = _to_grayscale_np(a)
    B = _to_grayscale_np(b)
    if A.shape != B.shape:
        from PIL import Image

        B = (
            np.array(
                Image.fromarray((B * 255).astype("uint8")).resize((A.shape[1], A.shape[0]), resample=Image.NEAREST),
                dtype=np.float64,
            )
            / 255.0
        )
    mu_a = A.mean()
    mu_b = B.mean()
    var_a = A.var()
    var_b = B.var()
    cov_ab = ((A - mu_a) * (B - mu_b)).mean()
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * cov_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    if den == 0:
        return 1.0
    return float(num / den)


def maybe_lpips(a: Any, b: Any) -> Optional[float]:
    """Compute LPIPS if the ``lpips`` package is installed; return
    None otherwise. The dispatcher treats None as "skipped"."""
    try:
        import lpips
        import torch
    except Exception:
        return None
    try:
        loss_fn = lpips.LPIPS(net="alex")
        loss_fn.eval()

        def _to_chw_tensor(im: Any) -> Any:
            import numpy as np
            from PIL import Image

            if hasattr(im, "convert"):
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            else:
                arr = np.asarray(im, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = (arr * 2 - 1).transpose(2, 0, 1)
            return torch.from_numpy(arr).unsqueeze(0)

        with torch.no_grad():
            score = float(loss_fn(_to_chw_tensor(a), _to_chw_tensor(b)).item())
        return score
    except Exception:
        return None


def extract_image_from_pytest_output(captured_output: str) -> Any:
    """Return a PIL.Image or None on miss."""
    from PIL import Image

    m = _IMAGE_MARKER_RE.search(captured_output)
    if m:
        after = captured_output[m.end() :]
        chunks: List[str] = []
        for line in after.splitlines():
            if not line.strip():
                break
            chunks.append(line.strip())
        payload = "".join(chunks)
        try:
            raw = base64.b64decode(payload, validate=False)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass

    pm = _IMAGE_PATH_RE.search(captured_output)
    if pm:
        path = Path(pm.group("path"))
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
    return None


@dataclass
class _DiffusionRef:
    image: Any
    prompt: str
    source_model_id: str = ""


class DiffusionComparator(Comparator):
    """Comparator for text-to-image diffusion models (SD 1.5 /
    SDXL / FLUX / SD3).

    Category claim: ``"Image"`` when the model has a "text-to-image"
    pipeline-tag indicator (encoded in the discriminator string).
    To avoid stealing the Image bucket from the ViT classification
    alias, the supports() check requires the model_id to contain a
    diffusion keyword.

    Phase 4: the comparator is registered but DOES NOT replace the
    classification comparator for plain "Image" category models.
    It only activates when the model_id clearly indicates a
    diffusion architecture.
    """

    category: str = "Image"
    discriminator: str = "diffusion"

    _DIFFUSION_KEYWORDS = (
        "stable-diffusion",
        "stable_diffusion",
        "sd-",
        "sdxl",
        "sd3",
        "flux",
        "dall-e",
        "kandinsky",
        "wuerstchen",
    )

    def supports(self, category: str, model_id: str) -> bool:
        if category != self.category:
            return False
        mid_l = model_id.lower()
        return any(k in mid_l for k in self._DIFFUSION_KEYWORDS)

    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        img = extract_image_from_pytest_output(captured_output)
        if img is None:
            return Evidence(
                payload=None,
                ok=False,
                reason=(
                    "could not find a generated image in the pytest "
                    "output. Expected '==IMAGE 0 - OUTPUT' marker "
                    "followed by base64 PNG bytes, OR an "
                    "'image: <path>.png' line referencing an "
                    "existing file."
                ),
            )
        return Evidence(
            payload=img,
            input_hint=None,
            ok=True,
            reason="image extracted from pytest output",
        )

    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> _DiffusionRef:
        """Generate the HF reference image on CPU. This is
        EXPENSIVE (5-30 minutes on CPU); guard with a timeout in
        the caller."""
        try:
            from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
            import torch
        except Exception as exc:
            raise RuntimeError(f"diffusion comparator: diffusers not installed " f"({type(exc).__name__})")
        prompt = evidence.input_hint or "a photograph of an astronaut riding a horse"

        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32)
        out = pipe(prompt, num_inference_steps=20)
        img = out.images[0] if hasattr(out, "images") else out
        return _DiffusionRef(image=img, prompt=prompt, source_model_id=model_id)

    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        if not isinstance(reference, _DiffusionRef):
            return ValidationResult(
                ok=False,
                reason="diffusion comparator: reference is not a _DiffusionRef",
            )
        a = evidence.payload
        b = reference.image
        ssim_score = ssim(a, b)
        lpips_score = maybe_lpips(a, b)

        ssim_pass = ssim_score >= DEFAULT_SSIM_MIN
        lpips_pass = lpips_score is None or lpips_score <= DEFAULT_LPIPS_MAX

        ok = ssim_pass and lpips_pass
        bits = [f"SSIM={ssim_score:.3f} (>= {DEFAULT_SSIM_MIN})"]
        if lpips_score is not None:
            bits.append(f"LPIPS={lpips_score:.3f} (<= {DEFAULT_LPIPS_MAX})")
        else:
            bits.append("LPIPS=skipped (lpips package not installed)")
        return ValidationResult(
            ok=ok,
            reason=("PASS: " if ok else "FAIL: ") + "; ".join(bits),
            tt_text="(image)",
            hf_text="(image)",
        )

    def build_repair_prompt(
        self,
        model_id: str,
        evidence: Evidence,
        result: ValidationResult,
        *,
        iter_idx: int,
        max_iters: int,
        previous_attempts: Optional[List[str]] = None,
        extra_blocks: Optional[Sequence[str]] = None,
    ) -> str:
        from .base import render_extra_blocks

        prev = "\n    ".join(previous_attempts or []) or "(none)"
        return (
            f"You are debugging a TT-hardware bring-up of {model_id!r} "
            f"(diffusion text-to-image). The generated image disagrees "
            f"with HF beyond the perceptual threshold.\n\n"
            f"  GATE VERDICT (iter {iter_idx}/{max_iters}):\n"
            f"    {result.reason}\n\n"
            f"  LIKELY SUSPECTS for diffusion bring-ups:\n"
            f"    1. UNet timestep embedding (sinusoidal vs learned).\n"
            f"    2. Cross-attention key/value projection from text.\n"
            f"    3. VAE decoder upsampling (latent stride 8 vs 16).\n"
            f"    4. Sampler schedule (DDIM vs Euler-A vs DPM-Solver).\n"
            f"    5. ClassifierFreeGuidance scale wired incorrectly.\n"
            f"    6. CLIP text-encoder pooling mismatch.\n\n"
            f"  WHAT WAS ALREADY TRIED:\n"
            f"    {prev}\n\n"
            f"  BUDGET: ~25 min/iter. Make at least one Edit.\n" + render_extra_blocks(extra_blocks)
        )


_singleton = DiffusionComparator()
register_comparator(_singleton)


__all__ = [
    "DEFAULT_LPIPS_MAX",
    "DEFAULT_SSIM_MIN",
    "DiffusionComparator",
    "extract_image_from_pytest_output",
    "maybe_lpips",
    "ssim",
]
