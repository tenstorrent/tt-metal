"""PyTorch ground-truth reference for Manifold-Constrained Hyper-Connections (mHC).

This is Deliverable 1 of tenstorrent/tt-metal#40703: a clear, unoptimised reference
that the tt-metal / tt-transformer kernels are validated against. Pure torch, fp32,
runs on CPU. Correctness over speed.

It mirrors the *real* DeepSeek-V4 inference code (deepseek-ai/DeepSeek-V4-Pro,
inference/model.py + inference/kernel.py), reproducing the exact math:

  - model.py  Block.hc_pre            -> MHCWrap.hc_pre        (reduce n streams -> 1)
  - model.py  Block.hc_post           -> MHCWrap.hc_post       (expand 1 -> n streams, mix residual)
  - kernel.py hc_split_sinkhorn_kernel-> parametrize/sinkhorn_knopp (compute the H matrices)
  - model.py  Transformer.forward     -> mhc_expand            (replicate embedding to n streams)
  - model.py  ParallelHead.hc_head    -> MHCHead               (collapse n streams -> 1 at the output)

----------------------------------------------------------------------------------------
mHC  vs  vanilla Hyper-Connections
----------------------------------------------------------------------------------------
The ONLY structural difference is the n x n stream-mixing matrix (called `comb` in
DeepSeek's code, H_res in the paper / design doc). mHC projects it onto the
*doubly-stochastic manifold* (Birkhoff polytope) via Sinkhorn-Knopp. Vanilla HC leaves
it unconstrained.

Doubly-stochastic  =>  largest singular value exactly 1 (NOT a matrix entry equal to 1), with the
all-ones vector a fixed eigenvector: the mean across streams is preserved, while off-mean
components are non-expansive (singular values <= 1) -- they may attenuate but never amplify.
The set is closed under matrix multiplication, so the depth-composite (product over all layers)
is *still* doubly stochastic at any depth -> it can never amplify the signal. That restores
the ResNet identity-mapping property that unconstrained HC loses. Set constraint="none"
to get vanilla HC and watch the depth-composite blow up (see the __main__ demo).

----------------------------------------------------------------------------------------
Naming map  (DeepSeek code  <->  paper / design-doc #40703)
----------------------------------------------------------------------------------------
  comb        H_res    n x n   stream mixing, doubly-stochastic        (applied as comb^T)
  pre         H_pre    1 x n   reduce: n streams -> 1 input for F
  post        H_post   n x 1   expand: F's output -> n streams
  fn          P        fused projection, stored [mix_hc, n*dim] (== P^T)
  base        b        bias    [mix_hc]
  scale       a        scalars [3] = (a_pre, a_post, a_res)
  mix_hc      n^2+2n   == (2+n)*n  raw outputs of the fused projection (24 for n=4)

The work splits into two components (per the design doc):
  * Parametrization  (X, P, b, a) -> H matrices       -> target: tt-metal C++ kernel
        == `parametrize()` here  (the Sinkhorn lives here)
  * Computation      (F, H matrices) -> X'            -> target: tt-transformer Python op
        == `MHCWrap.hc_pre` / `hc_post` here
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MHCConfig:
    """mHC hyperparameters. Defaults are the DeepSeek-V4-Pro config.json values."""

    dim: int = 7168  # C  -- model hidden size                ("dim")
    n: int = 4  # expansion rate / number of streams     ("hc_mult")
    sinkhorn_iters: int = 20  # Sinkhorn-Knopp iterations              ("hc_sinkhorn_iters")
    eps: float = 1e-6  # stochasticity epsilon                  ("hc_eps")
    norm_eps: float = 1e-6  # RMSNorm denominator epsilon            ("norm_eps")

    @property
    def mix_hc(self) -> int:
        # n (pre) + n (post) + n*n (comb) == (2 + n) * n == n^2 + 2n
        return (2 + self.n) * self.n


# ----------------------------------------------------------------------------------------
# Parametrization: X, P, b, a  ->  H matrices  (the "compute the H matrices" stage)
# ----------------------------------------------------------------------------------------
def sinkhorn_knopp(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Project a batch of n x n matrices onto the doubly-stochastic manifold.

    Exactly mirrors DeepSeek-V4 kernel.py:hc_split_sinkhorn_kernel:
      1. row softmax (+eps)  -- strictly positive init, rows ~sum to 1
      2. one column normalisation
      3. (iters - 1) alternations of row- then column-normalisation
    Ends on a column normalisation: columns sum to 1 exactly, rows to 1 within ~eps.

    `logits`: [..., n, n] -> doubly-stochastic [..., n, n].

    Note on reduction axes: dim=-1 sums within a row (row sums); dim=-2 sums within a
    column (column sums) -- matching the kernel's dim=1 / dim=0 reductions.
    """
    m = torch.softmax(logits, dim=-1) + eps  # row softmax init
    m = m / (m.sum(dim=-2, keepdim=True) + eps)  # column normalise
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)  # row normalise
        m = m / (m.sum(dim=-2, keepdim=True) + eps)  # column normalise
    return m


def parametrize(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    cfg: MHCConfig,
    constraint: str = "sinkhorn",
):
    """Split the fused projection output into the three H matrices.

    Mirrors kernel.py:hc_split_sinkhorn_kernel (the split + affine + constraints).

    `mixes`: [..., mix_hc]   raw projection output  ( == RMSNorm(X) @ P )
    `scale`: [3]             (a_pre, a_post, a_res)
    `base`:  [mix_hc]        bias b
    `constraint`:
        "sinkhorn" -> mHC        (doubly-stochastic H_res)            <-- the real thing
        "none"     -> vanilla HC (unconstrained H_res; contrast only)

    Returns (pre [..., n], post [..., n], comb [..., n, n]).
    """
    n, eps = cfg.n, cfg.eps
    pre_raw = mixes[..., 0:n]
    post_raw = mixes[..., n : 2 * n]
    comb_raw = mixes[..., 2 * n :].reshape(*mixes.shape[:-1], n, n)

    b_pre = base[0:n]
    b_post = base[n : 2 * n]
    b_comb = base[2 * n :].reshape(n, n)

    pre = torch.sigmoid(pre_raw * scale[0] + b_pre) + eps  # H_pre   in (eps, 1+eps)
    post = 2.0 * torch.sigmoid(post_raw * scale[1] + b_post)  # H_post  in (0, 2)
    comb_logits = comb_raw * scale[2] + b_comb  # H_res pre-constraint

    if constraint == "sinkhorn":
        comb = sinkhorn_knopp(comb_logits, cfg.sinkhorn_iters, eps)  # mHC manifold projection
    elif constraint == "none":
        comb = comb_logits  # vanilla HC (unconstrained)
    else:
        raise ValueError(f"unknown constraint {constraint!r}")
    return pre, post, comb


# ----------------------------------------------------------------------------------------
# Computation: apply the H matrices around a sublayer F  (the "single-layer propagation")
# ----------------------------------------------------------------------------------------
class MHCWrap(nn.Module):
    """Wrap one sublayer F with mHC. One instance == one mHC-wrapped sublayer.

    DeepSeek-V4 has TWO of these per transformer block: one around attention and one
    around the MoE FFN, each with independent parameters.

    Trainable parameters (constant at inference):
        fn:    [mix_hc, n*dim]  fused projection P  (DeepSeek hc_attn_fn / hc_ffn_fn)
        base:  [mix_hc]         bias b              (DeepSeek hc_attn_base / hc_ffn_base)
        scale: [3]              scalars a           (DeepSeek hc_attn_scale / hc_ffn_scale)
    The H matrices themselves are NOT stored -- they are recomputed from X every forward
    pass (dynamic), which is the whole point of hc_pre below.
    """

    def __init__(self, cfg: MHCConfig, constraint: str = "sinkhorn"):
        super().__init__()
        self.cfg = cfg
        self.constraint = constraint
        self.fn = nn.Parameter(torch.empty(cfg.mix_hc, cfg.n * cfg.dim))
        self.base = nn.Parameter(torch.zeros(cfg.mix_hc))
        self.scale = nn.Parameter(torch.full((3,), 0.01))  # paper: alpha init ~ 0.01
        nn.init.normal_(self.fn, std=0.02)

    def hc_pre(self, x: torch.Tensor):
        """Reduce n streams -> 1 and produce post/comb for the matching hc_post.

        Mirrors model.py:Block.hc_pre.  x: [b, s, n, d] -> y: [b, s, d].
        The projection input is RMS-normalised (no learned weight); the actual stream
        reduction uses the raw stream values.
        """
        cfg = self.cfg
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()  # [b, s, n*d]
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
        mixes = F.linear(xf, self.fn) * rsqrt  # == RMSNorm(xf) @ fn^T -> [b, s, mix_hc]
        pre, post, comb = parametrize(mixes, self.scale, self.base, cfg, self.constraint)
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)  # weighted sum of streams -> [b, s, d]
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand F's output 1 -> n streams and mix the residual streams via comb (H_res).

        Mirrors model.py:Block.hc_post.  Per output stream j:
            new_j = post_j * F(.)  +  sum_i comb[i, j] * residual_i
        i.e. the residual mixing uses comb^T (still doubly-stochastic).
        x: [b, s, d], residual: [b, s, n, d] -> [b, s, n, d].
        """
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        # torch.sum is the term 1 H_res * x_l
        #
        return y.type_as(residual)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """x: [b, s, n, d]; `sublayer`: callable [b, s, d] -> [b, s, d].

        The sublayer owns its own pre-norm, exactly like DeepSeek's Block (which does
        attn_norm(x) then attn(x) between hc_pre and hc_post).
        """
        residual = x
        h, post, comb = self.hc_pre(x)
        h = sublayer(h)
        return self.hc_post(h, residual, post, comb)


def mhc_expand(h: torch.Tensor, n: int) -> torch.Tensor:
    """Embedding [b, s, d] -> n identical residual streams [b, s, n, d].

    Mirrors model.py:Transformer.forward (h.unsqueeze(2).repeat(1, 1, n, 1)).
    """
    return h.unsqueeze(2).repeat(1, 1, n, 1)


class MHCHead(nn.Module):
    """Collapse the n streams back to 1 at the output, before the final norm + lm_head.

    Pre-only (no post/comb) because we leave the residual highway here.
    Mirrors model.py:ParallelHead.hc_head.  Params: fn [n, n*dim], base [n], scale [1].
    """

    def __init__(self, cfg: MHCConfig):
        super().__init__()
        self.cfg = cfg
        self.fn = nn.Parameter(torch.empty(cfg.n, cfg.n * cfg.dim))
        self.base = nn.Parameter(torch.zeros(cfg.n))
        self.scale = nn.Parameter(torch.full((1,), 0.01))
        nn.init.normal_(self.fn, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()  # [b, s, n*d]
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
        mixes = F.linear(xf, self.fn) * rsqrt  # [b, s, n]
        pre = torch.sigmoid(mixes * self.scale + self.base) + cfg.eps
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)  # [b, s, d]
        return y.to(dtype)


# ----------------------------------------------------------------------------------------
# Demo + invariant checks (run:  python mhc_reference.py)
# ----------------------------------------------------------------------------------------
def _demo() -> None:
    torch.manual_seed(0)
    # Small dim so the demo is instant on CPU; the mHC math is independent of dim.
    # Real DeepSeek-V4: dim=7168, n=4, sinkhorn_iters=20.
    cfg = MHCConfig(dim=64, n=4, sinkhorn_iters=20)
    b, s = 2, 8

    print(f"config: dim={cfg.dim} n={cfg.n} mix_hc={cfg.mix_hc} sinkhorn_iters={cfg.sinkhorn_iters}")

    # --- end-to-end shape sanity: embed -> expand -> 1 mHC-wrapped block -> head ---
    embed = torch.randn(b, s, cfg.dim)
    h = mhc_expand(embed, cfg.n)
    assert h.shape == (b, s, cfg.n, cfg.dim)

    attn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
    attn = nn.Linear(cfg.dim, cfg.dim)  # stand-in for F (attention / MoE)
    wrap = MHCWrap(cfg, constraint="sinkhorn")
    h = wrap(h, lambda z: attn(attn_norm(z)))  # sublayer owns its pre-norm
    assert h.shape == (b, s, cfg.n, cfg.dim)
    out = MHCHead(cfg)(h)
    assert out.shape == (b, s, cfg.dim)
    print(f"shapes OK: embed {tuple(embed.shape)} -> streams {(b, s, cfg.n, cfg.dim)} -> head {tuple(out.shape)}")

    # --- invariant 1: comb (H_res) is doubly-stochastic ---
    _, _, comb = wrap.hc_pre(mhc_expand(torch.randn(b, s, cfg.dim), cfg.n))
    row = comb.sum(-1)  # should be ~1
    col = comb.sum(-2)  # should be ~1 (exact: last Sinkhorn step is a column norm)
    print(
        f"\n[mHC] H_res doubly-stochastic: "
        f"max|rowsum-1|={(row - 1).abs().max():.2e}  max|colsum-1|={(col - 1).abs().max():.2e}  "
        f"min entry={comb.min():.2e} (>=0)"
    )
    one = comb[0, 0]
    svals = torch.linalg.svdvals(one)
    print(f"       one H_res spectral norm (sigma_max) = {svals[0]:.6f}  (doubly-stochastic => exactly 1)")

    # --- invariant 2: closed under multiplication ---
    a2, b2 = comb[0, 0], comb[0, 1]
    prod = a2 @ b2
    print(
        f"[mHC] closure: product of two H_res is doubly-stochastic: "
        f"max|rowsum-1|={(prod.sum(-1) - 1).abs().max():.2e}  max|colsum-1|={(prod.sum(-2) - 1).abs().max():.2e}"
    )

    # --- invariant 3: identity mapping across DEPTH (mHC) vs blow-up (vanilla HC) ---
    # Depth-composite is P_L = comb_L^T @ ... @ comb_1^T. Doubly-stochastic => ||P_L||_2 = 1
    # at every depth and P_L @ 1 = 1 (mean across streams preserved). Unconstrained => exponential drift.
    print("\nDepth-composite spectral norm ||P_L||_2  (the residual 'highway' across L layers):")
    print(f"{'L':>5} | {'mHC (doubly-stochastic)':>24} | {'vanilla HC (unconstrained)':>26}")
    print("-" * 62)
    n = cfg.n
    P_mhc = torch.eye(n)
    P_van = torch.eye(n)
    uniform = torch.full((n, n), 1.0 / n)  # the pure 'mean' operator
    checkpoints = {1, 5, 20, 61, 200}
    dev_at_end = None
    for layer in range(1, 201):
        # mHC: a genuine doubly-stochastic matrix from the Sinkhorn path
        comb_mhc = sinkhorn_knopp(torch.randn(n, n), cfg.sinkhorn_iters, cfg.eps)
        # vanilla HC: an arbitrary learned n x n matrix (here ~N(0, 0.5)), no manifold constraint
        comb_van = torch.randn(n, n) * 0.5
        P_mhc = comb_mhc.T @ P_mhc
        P_van = comb_van.T @ P_van
        if layer in checkpoints:
            print(
                f"{layer:>5} | {torch.linalg.svdvals(P_mhc)[0].item():>24.4f} | "
                f"{torch.linalg.svdvals(P_van)[0].item():>26.4e}"
            )
        if layer == 200:
            dev_at_end = (P_mhc - uniform).abs().max().item()
    print(
        f"\n[mHC] stays ~1 at all depths (bounded, signal preserved); also P_200 @ 1 - 1 = "
        f"{(P_mhc @ torch.ones(n) - 1).abs().max():.2e} (mean across streams preserved)."
    )
    print(f"[mHC] P_200 has converged toward the uniform mean operator J/n: max|P_200 - J/n| = {dev_at_end:.2e}")
    print("[vanilla HC] ||P_L||_2 drifts exponentially with depth -> the identity mapping is lost.")


if __name__ == "__main__":
    _demo()
