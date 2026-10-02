## Setup 'chippy'

`git clone git@yyz-gitlab.local.tenstorrent.com:syseng-platform/chippy.git`

Log into one of the soc machines (e.g. `soc-l-#`) or another machine with GGC 13 available.

```
source /tools_soc/tt/bin/bashrc
module load cmake/3.30.3
module load gcc/13
cd $CHIPPY_DIR

cmake -S lib -B ../chippy-build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG" \
  -DCHIPPY_BUILD_UNIT_TESTS=OFF \
  -DBUILD_EXCLUDE_JLINK=TRUE \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build $CHIPPY_DIR/../chippy-build --target grendel
```

## Build tt-metal

Need to add additional defines to build. After building the first time, can just run usual `./build_metal.sh` commands unless the CMAKE files are removed (e.g. with a `--clean` or `git clean`).

```
./build_metal.sh --build-tests --build-umd-tests --configure-only

cmake -DTT_UMD_BUILD_GRENDEL_JTAG=ON \
  -DCHIPPY_SOURCE_DIR=$CHIPPY_DIR \
  -DCHIPPY_BUILD_DIR=$CHIPPY_DIR/../chippy-build \
  -B build

cmake --build build --target install
```

*NOTE:* Probably don't need this, but keeping it here in case I'm wrong.
`  -DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12 \`


## Launch Emulation Server

`git clone git@yyz-gitlab.local.tenstorrent.com:tensix/soc/grendelemulation.git`

1. Log into one of the soc machines (e.g. `soc-l-#`)
2. Setup your environment. Choose either the `mimir` directory (single mimir chiplet - 1 emulation module) or `mmk` (2 mimir chiplets + 1 keraunos - 4 emualtion modules) directory depending on your needs.
```
source /tools_soc/tt/bin/bashrc
cd $GRENDELEMU_DIR/models/mimir
source bin/setup_env.sh
```
3. Launch the emulation server
```
emu run -t 3600 -- -sv tests/test_sival_server.py --disable-dpi
```

**NOTE:** Sometimes just re-launching the server doesn't work properly (it exits immediately thinking there is no design) and the setup commands must be re-run before re-launching the server.

**NOTE:** If you are switching between emulation models, sometimes the previous setup leaves behind paths that cause failures. To be safe, start from a fresh terminal whenever you switch emulation models.

**NOTE:** Both chippy & tt-metal will need to know the server host & port to connect to the emulation server. You will see host listed several times in the output. The default port is `8080` for emulation (and is explicit in the last example below). Note the host & port for future steps.
`Running on host soc-zebu-01`
`Running on emulation host: soc-zebu-01`
`INFO     server._start_server        SiVal server bound to soc-zebu-01:8080`

## Use 'chippy' to load firmware

There are 3 FW bring-up scripts:
- `bringup_mimir_1x1.py` -> Requires at least 1 Mimir chiplets. Brings up 2 CCEs & connections to adjacent DRAM.
- `bringup_two_mimir.py` -> Requires at least 2 Mimir chiplets connected to each other. Brings up 2 CCEs in each mimir, connections to adjacent DRAM from each mimir & D2D connections between the mimirs.
- `bringup_mmk.py` -> Requires at least 2 Mimir chiplets connected to each other & connected to 1 Keraunos. Brings up CCEs for all 3 chiplets, connections to adjacent DRAM from each mimir & D2D connections between all 3 chiplets.

Command:
```
cd $CHIPPY_DIR
uv run --project validation python validation/metal_bringup/<script> <emu_host> <emu_port>
```

## Run tt-metal test

Choose between `mimir_1x1.yaml` & `mimir_x2_package.yaml` depending on needs.

```
export TT_METAL_HOME=<dir>
export TT_METAL_EMU_SERVER=<host>:<port>
export TT_METAL_EMU_SOC_DESC=$TT_METAL_HOME/tt_metal/third_party/umd/tests/soc_descs/<yaml_file>
```

Run Mimir tests. E.g.
`TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_context --gtest_filter=*MimirEmu*`
