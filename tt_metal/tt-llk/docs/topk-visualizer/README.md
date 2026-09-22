# Top-K Kernel Lab

An animated, dependency-free website for the ordinary non-stable Blackhole Top-K network in `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h`.

Run from this directory:

```sh
python3 server.py --port 8765
```

Open http://localhost:8765. The server defaults to loopback. In a remote workspace, forward port 8765 to your browser. No build step, npm packages, external fonts, or runtime CDN is required.

The example is a complete 32×64 matrix in two value tiles, using K=32. Select any of the eight faces to see all 256 values in its actual DEST row/column layout. Every displayed number is one datum. The default numbered input uses `value = row * 64 + column`, so the initial unpack transpose is easy to follow.

- Follow any original matrix row: it starts horizontally in the L1 face and becomes a DEST column after unpacking. Face placement also transposes: input faces `[F0, F1, F2, F3]` become `[F0ᵀ, F2ᵀ, F1ᵀ, F3ᵀ]`.
- All 32 lanes are drawn in each of LREG0–3. Choose an LREG to highlight the 32 DEST positions read by its load: four consecutive rows and eight even or odd columns. The four green lanes in each register belong to the original row being followed; the other 28 lanes contain actual values from seven other rows.
- DEST addresses are logical ISA row addresses relative to the value region. Tile 0 faces start at rows 0, 16, 32 and 48; tile 1 adds 64. They are not L1 byte addresses.
- **Guided** expands the first working sets and summarizes repeated loads. **Every step** exposes all 329 modeled operations for the selected row group. Load/store events group the helper's vector instructions.
- Step, play, change speed, scrub, jump by chapter or milestone, select a value to follow its original index, and try shuffled, signed, or descending inputs.
- Space plays or pauses; arrow keys step; Home/End jump to the beginning/result when focus is outside controls.
- The source panel links each operation to the current local header or caller. The server only exposes the website directory and the two allowlisted source files.
- The result names the selected original row and its input range. Its row selector changes the 32 winners shown; expand the complete output to inspect all 32×32 winners. JSON export includes the original matrix, full output value/index matrices, and selected row's trace.

The model executes the register compare/exchange and transpose schedule independently for all 32 rows; it does not use a CPU sort to produce the result. The stepper follows one of the driver's four face/column passes: even rows 0–14, odd rows 1–15, even rows 16–30, or odd rows 17–31. Earlier passes show their completed state for the current entry point, and later passes show its input state. Other passes, instruction timing, replay recording, address-counter instructions and synchronization are summarized.

Index payloads are carried throughout and available in the value inspector; their separate DEST tiles and LREG4–7 are not drawn. Registers marked “not active” omit stale hardware contents. Input values are distinct within each row and exactly representable in FP32. Stable, fused, rank-stamped, UInt16-value and special-value paths are outside this lesson.

For K=32, the source's rebuild default branch declares an inner `total_datums_to_compare=64`, so this model visits both halves despite the caller requesting `skip_second=1`. Only the first half is returned. Unpack/pack orchestration comes from `tests/sources/topk_test.cpp`, outside the SFPU header.

Validate the algorithm with Node 18 or later:

```sh
node engine.test.mjs
node layout.test.mjs
```

The checks cover 201 complete algorithm traces plus all 2048 matrix-to-DEST coordinates, full-lane load/store footprints and transposes, row-group ordering, and complete matrix outputs. No accelerator is required or exercised.

Browser smoke checks (optional Playwright installation):

```sh
python3 browser_test.py
```

Hardware references used to check the projection:

- [Blackhole SFPLOAD](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md)
- [Blackhole SFPSTORE](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTORE.md)
- [Blackhole SFPSWAP](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md)
- [SFPTRANSP (shared by Blackhole and Wormhole)](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SFPTRANSP.md)
