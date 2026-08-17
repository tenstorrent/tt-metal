<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# axes

One file per op, naming which parameter of the op's codegen sweep carries the shape, the dtype and
the layout, and which sweep parameter supplies each keyword argument of the native call.

This is the only thing about a port that is still written down. Everything else `discover.py` reads
off the generator tree, the tt-metal tree or the built ttnn module, and verifies. This cannot be
derived, because a sweep names its parameters however it likes: `dtype` in `codegen_move`,
`input_a_dtype` in `codegen_tilize`. Some bundle several fields into one dict-valued parameter, so a
name may be a dotted path — `ri_specs.shape` descends into the value of the `ri_specs` parameter.

These live here, in tt-metal, beside the harness that reads them. They used to be the `vector_map`
field of `agentic_port/manifests/<op>.yaml` in tt-dm-codegen, which is a sibling porting pipeline's
internal state file on a non-default branch. Six lines an op does not justify that coupling.

## Format

```yaml
shape: input_shape          # sweep parameter carrying the input shape
dtype: input_a_dtype        # sweep parameter carrying the input dtype
layout: input_a_layout      # sweep parameter carrying the input layout
kwargs:                     # native kwarg name -> sweep parameter supplying it
  memory_config: output_memory_config
suites: [nightly]           # which keys of the sweep's `parameters` dict to enumerate
```

`suites` may be omitted, in which case every suite the sweep defines is enumerated. Listing several
takes their union, deduplicated on the input signature, because suites overlap — `untilize` draws
tile-aligned `bfloat8_b` from `broaden_suite` and everything else from `nightly`.

## Optional narrowing

Three more blocks are accepted, all optional and all absent for most ops. Four of the seven ops
declare none of them, `tilize` included.

```yaml
scope:                      # the ported entry point is narrower than the op
  layouts: [tile]
  dtypes: [bfloat16, bfloat8_b]
  tile_aligned: [bfloat8_b] # these dtypes only on a whole number of tiles
ungradeable_reasons:        # sweep rejections that are op limitations, not codegen ones
  - "TILE front-padding not supported in codegen"
bad_golden:                 # slices where the oracle itself is wrong
  - {dtype: float32, layout: row_major, nonzero_kwarg: value, reason: "..."}
```

A point outside `scope` is still valid for the op, so it becomes a routing check rather than being
dropped — the public op must fall back to native there, and that is what the emitted routing test
asserts. Those `reason` strings ship in the public diff, so write them as the constraint rather than
as the mechanism.

`scope` is a narrowing of last resort, and it is worth knowing that it is already close to
redundant: `gate.py` grades against the prototype leg, so a case the generator itself cannot serve is
excused whether or not anything declares it. What `scope` still buys is the routing assertion and the
human-readable reason, neither of which a measurement produces on its own.
