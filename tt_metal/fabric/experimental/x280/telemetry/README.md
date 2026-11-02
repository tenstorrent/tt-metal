# X280 Telemetry

Experimental firmware that runs on hart 0 of one X280 tile per chip, reads ARC
telemetry and writes it into a table in LIM. Two host programs go with it:

- `x280_telemetry_loader` primes the chips, or boots resident idle if needed and
  loads the telemetry image.
- `x280_telemetry_app` reads the table from each chip and prints it.

## Building

Run everything from the repo root.

1. Configure (fresh checkout only):

   ```bash
   ./build_metal.sh --build-metal-tests --configure-only
   cmake -B build -DTT_METAL_BUILD_X280=ON
   ```

2. Build the firmware and both host programs:

   ```bash
   ninja -C build x280_image_firmware-telemetry x280_telemetry_loader x280_telemetry_app
   ```

   After changing only `telemetry.c`, rebuild just the image:

   ```bash
   ninja -C build x280_image_firmware-telemetry
   ```

## Running

1. Set the environment:

   ```bash
   export TT_METAL_RUNTIME_ROOT=$PWD
   ```

   If Metal can't place the board (for example, only one chip of a P300 is
   visible), also point it at a mesh graph descriptor:

   ```bash
   export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
   ```

2. Prime, once per device. If it reports that a chip was primed, reset that chip
   with `tt-smi -r <id>`:

   ```bash
   build/tt_metal/fabric/experimental/x280/telemetry/x280_telemetry_loader --prime
   ```

3. Load and run the firmware. It prints "Firmware running" for each chip:

   ```bash
   build/tt_metal/fabric/experimental/x280/telemetry/x280_telemetry_loader
   ```

   An optional `image.bin` argument loads a different image.

4. Read and print the tables. It prints "Table for chip: N" followed by that
   chip's rows:

   ```bash
   build/tt_metal/fabric/experimental/x280/telemetry/x280_telemetry_app
   ```

The loader stops the firmware right after it starts, so the app reads the last
values the firmware wrote, not live ones. Neither program resets a chip.
