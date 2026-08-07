# Fused band op — design, grounded in where the 53 ops actually come from

Written before the implementation, because the op-count structure turned out not to be what
`AUDIO_FUSION_PLAN.md` assumed, and it changes the op's shape.

## Where the 53 ops per band come from

The plan treated the band as "8 convolutions plus 45 scaffolding ops". The profile disagrees about
what the scaffolding *is*. From PROFILE_2026_08_06.txt, over 127 bands:

    HaloDeviceOperation      870      Conv2dDeviceOperation   870      MoveDeviceOperation   870
    PaddedSliceDeviceOp      654      SliceWriteDeviceOp      654

**Halo, Move and Conv2d are all exactly 870.** One Halo and one Move per conv *invocation*, and
870 / 127 = **6.85 conv invocations per band — where the band contains only two convolutions.**

The reason is already written down in `audio_ops.py:375`: we hand `ttnn.conv1d` a DRAM-interleaved
tensor with no slice config, so every call takes the DRAM slicing loop, and that loop emits

    PaddedSlice -> Halo -> Move -> Conv2d -> SliceWrite

per slice. At ~3.4 slices per convolution that is 2 x 3.4 x 5 = ~34 ops, which is most of the 53.

So the band is not "2 convs surrounded by scaffolding". It is **2 convs multiplied ~3.4x by DRAM
slicing, each slice paying a 5-op wrapper**, plus the activation, its layout round trip, and the pads.

## What that implies for the op

The fused op cannot be one invocation per band -- the data does not fit L1, which is why the slicing
exists. It must be **one invocation per T-chunk**, each chunk doing the whole band:

    per chunk:  read window -> upsample x2 -> snake -> downsample x2 -> write

At the same ~3.4 chunks per band that is **~3.4 ops per band instead of ~53**, which is the 4-5x, and
it is the same arithmetic the plan reached by a different and partly wrong route. The win comes from
the wrapper collapsing: one PaddedSlice/Halo/Move/SliceWrite set per chunk instead of one per conv per
chunk, and no activation op or layout round trip at all.

## The math the kernel has to do

Every stage is per-channel, so the op is a 1-D nonlinear stencil applied independently per channel.
For ratio 2 with `up_kernel_size = down_kernel_size = K = 12`:

* **Upsample** is polyphase, so no zero-stuffed tensor is ever built. With taps `h[0..K-1]` scaled by
  the ratio, the two phases are the even and odd tap subsets, each length K/2 = 6:

      u[2n]   = sum_j h[2j]   * x[n - j + off]
      u[2n+1] = sum_j h[2j+1] * x[n - j + off]

* **Snake**, pointwise on the upsampled stream, with per-channel `alpha` and `inv_beta = 1/(beta+eps)`
  precomputed on the host so the kernel needs no reciprocal:

      s = u + inv_beta * sin(alpha * u)^2

* **Downsample** is a stride-2 FIR over `s`, so only even outputs are needed and only those are
  computed:

      y[n] = sum_k g[k] * s[2n - k + off]

Composing them, one output `y[n]` depends on a window of about `K/2 + K/2 = 12` input samples. The
kernel never materialises `u` or `s` beyond the small window it needs, which is the entire point --
those two intermediates are what currently cost a DRAM round trip each.

**The trap, carried over from `audio_resample.py::_forward_fused` and still true here:** replicate
padding does *not* decompose into per-phase replicate padding, because the pad region is a constant
whose parity alternates. The chunk edges must take their halo from real neighbouring samples, and only
the band's outermost chunks apply the replicate rule.

## Why this reuses the existing op rather than being a new one

A new ttnn op needs device operation, program factory, reader/writer/compute kernels, pybind and
CMake, then a full rebuild (~40 min, measured on the h3-prefix build). Most of that already exists and
works for `conv1d` depthwise: the halo gather, the sharding, the slicing loop, the weights CB, and --
from this branch -- SFPU fp32 tap accumulation and a fused-activation seam in
`compute_depthwise_conv1d.cpp`.

So the cheaper route is to extend that kernel with a band mode: same reader, same halo, same weights
CB carrying both tap sets, and a compute kernel that runs the three stages in DST before packing. The
per-channel snake parameters are the one genuinely new host-side piece, and they are the same
optional-input-tensor problem Step 2a already scoped.

## Cost, honestly

Step 2a was priced at ~20-30 ms (`fuse_saving.py`) and is *not* where the 5x is. This op is, because
it removes the slicing wrapper rather than one activation. But the projection rests on a fused
invocation costing about what a conv invocation costs today, and per `op_floor.py` the floor is 180 us
regardless of work, so that assumption is better supported here than anywhere else in this plan.

Expected: ~53 ops/band -> ~4, i.e. ~6700 ops -> ~500 for the band portion, against a measured floor of
180 us/op. That is the only remaining route to 60 ms.
