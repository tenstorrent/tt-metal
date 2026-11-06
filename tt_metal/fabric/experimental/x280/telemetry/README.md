# X280 Telemetry

Experimental firmware that runs on hart 0 of one X280 tile per chip. It reads
ARC telemetry every 2 seconds and writes it into a table in LIM at `0x08132000`.

`x280_telemetry_tui` is the host program that goes with it. It primes the chips,
or loads the firmware, keeps it running, and shows each chip's table live in a
terminal UI built with [FTXUI](https://github.com/ArthurSonzogni/FTXUI).

## Building

Run everything from the repo root.

1. Configure (fresh checkout only). CMake fetches FTXUI v6.1.9 during this step:

   ```bash
   ./build_metal.sh --build-metal-tests --configure-only
   cmake -B build -DTT_METAL_BUILD_X280=ON
   ```

2. Build the TUI. This also builds the firmware image it loads,
   `build/x280/firmware-telemetry2.bin`:

   ```bash
   ninja -C build x280_telemetry_tui
   ```

   After changing only `telemetry_v2.c`, rebuild just the image:

   ```bash
   ninja -C build x280_image_firmware-telemetry2
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

2. Prime, once per device. It prints "All chips are primed." when there's
   nothing to do. If it reports that a chip needs a reset, run `tt-smi -r <id>`:

   ```bash
   build/tt_metal/fabric/experimental/x280/telemetry/x280_telemetry_tui --prime
   ```

3. Start the UI:

   ```bash
   build/tt_metal/fabric/experimental/x280/telemetry/x280_telemetry_tui
   ```

   It opens the devices, boots resident idle where needed, loads the firmware
   on the first X280 tile of each chip and waits for it to run, then takes over
   the terminal.

   Options:

   - `--poll-ms N`: how often to read the tables, default 1000.
   - `image.bin`: load a different image instead of `firmware-telemetry2.bin`.

The program never resets a chip.

## Using the UI

Each chip gets a tab in the header with its status:

| Status | Meaning |
| --- | --- |
| LIVE | The table is published and the ARC heartbeat is changing. |
| WAITING | The firmware is running but hasn't published the table yet. |
| STALE | The ARC heartbeat hasn't changed for 10 seconds, so ARC or the firmware is stuck. |
| STOPPED | Hart 0 isn't running the image. |
| READ ERROR | The last table read failed; older data is still shown. |
| NO TILE | The tile couldn't be observed. |

Views:

- Dashboard (`1`): thermals, power and clocks, with gauges against their ARC
  limits; graphs of ASIC temperature, TDP and AICLK; and a firmware panel
  with the image's version, hart 0's phase and stage, the ARC heartbeat, read
  counts and the board's firmware versions.
- All metrics (`2`, or `v` to toggle): every row of the table, decoded and raw.
  Values that changed in the last read are highlighted.

Keys:

| Key | Action |
| --- | --- |
| `←` `→`, Tab, `h` `l` | Switch chip |
| `1` `2` `v` | Switch view |
| `↑` `↓`, `j` `k`, PgUp PgDn, Home End | Scroll the metrics |
| `q`, Esc, Ctrl-C | Quit |

Quitting stops the firmware and returns the tiles to resident idle; the program
prints "All tiles returned to resident idle." on exit. `kill` (SIGTERM) does the
same. Don't use `kill -9`: it leaves the firmware running on the tile.

Values are decoded the way UMD decodes Blackhole telemetry
(`firmware_info_provider_implementation.cpp` and `firmware_utils.cpp` under
`tt_metal/third_party/umd`). Tags UMD doesn't decode are shown raw.

## Logs

Metal's log would draw over the UI, so the TUI sends it to
`x280_telemetry_tui.log` in the current directory. Set `TT_LOGGER_FILE` to use
another file. `--prime` logs to the terminal as usual.

## Files

| Path | What it is |
| --- | --- |
| `telemetry_v2.c` | The firmware: hart 0 reads ARC and fills the table until STOP. |
| `common/telemetry_table_v2.h` | Table layout, shared by the firmware and the host. |
| `host/telemetry_tui.cpp` | The terminal UI. |
| `host/telemetry_host_lib.cpp`, `.hpp` | Loads and stops the firmware, and reads the table. |
| `host/legacy/` | Earlier host programs, no longer built. |
| `telemetry.c`, `common/telemetry_table.h` | The v1 firmware and table, no longer built. |

## Legacy programs

`host/legacy/` holds the programs from before the TUI: `x280_telemetry_loader`,
`x280_telemetry_app` and `x280_telemetry_host_app`. Their targets, and the v1
`firmware-telemetry` image, are commented out in `CMakeLists.txt`; uncomment a
block to build one again.
