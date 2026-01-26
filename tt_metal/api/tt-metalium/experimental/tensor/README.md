# Tensor in Runtime

This folder hosts the current effort on lowering Tensor concepts from TTNN to TT-Metal.

The goal is to make Tensor a first-class citizen in Metal Runtime while exposing a reasonable level of abstraction.

## Why experimental

Tensor migration represents work with significant transient states.

This folder
- allows Runtime to continue the effort with arbitrary steps and minimal distraction to other functions.
- unblocks other efforts interested in Host/Device Tensor but not the TTNN migration
- allows experimentation without the restrictions of our deprecation policy.

## namespace

Headers in this folder are not in the experimental folder as this staging area is meant to be short lived,
most of the concepts hosted are well tested, production code already.

## Life-time

This folder is expected to be short-lived, the effort is tracked by:
https://github.com/tenstorrent/tt-metal/issues/36373
