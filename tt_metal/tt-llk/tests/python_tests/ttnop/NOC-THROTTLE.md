# Wire Quasar NoC throttling into ttnop

LOCAL ONLY. `noc_throttle.py` and this file are untracked. Do not commit them.
When the user asks to add NoC throttling to ttnop, follow this file exactly.
do not invent a second mechanism, a kernel-side write, or a per-variant poke.

## What this is

A second perturbation setting, independent of the fillers. A filler moves one
Tensix thread against another. This moves every NoC transfer against all of
them, by holding this tile's NIU to one handshake per N cycles.

`TTNOP_NOC_THROTTLE=256` is 1/256. There is no kernel change and no rebuild.
the host pokes two Quasar `NOC_CFG` registers through the same
`read_word_from_device` / `write_words_to_device` seam ttnop already uses for
the cave.

Quasar only. Wormhole and Blackhole have no throttler registers. Asking for
one there must be an error, not a silent no-op.

## Hardware (do not re-derive)

Addresses from

- `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h`
  `NOC_CFG(cnt) = TT_NOC_REG_MAP_BASE_ADDR + 0x100 + cnt * 4`
- `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/tt_tensix_noc_overlay_reg.h`
  `TT_NOC_REG_MAP_BASE_ADDR = 0x02000000`
- cfg ids in `noc_parameters.h`: `THROTTLER_CYCLES_PER_WINDOW = 0xb`,
  `THROTTLER_HANDSHAKES_PER_WINDOW_NIU = 0xc`

So the two registers are:

| name | cfg id | address | write for 1/N |
| --- | --- | --- | --- |
| `THROTTLER_CYCLES_PER_WINDOW` | `0xb` | `0x0200012C` | `N` |
| `THROTTLER_HANDSHAKES_PER_WINDOW_NIU` | `0xc` | `0x02000130` | `1` |

One shared window feeds five per-port handshake limits: NIU, north, east,
south, west (`0xc` through `0x10`). Only the NIU one is programmed here. It is this
tile's own injection into the NoC. Do not also write N/E/S/W unless the user
asks. That would throttle the whole router, not just this core.

There is a second, per-VC throttler (`VC_THROTTLER_*`, cfg ids `0x2C` through `0x30`).
Do not use it. Its `CYCLES_PER_WINDOW` field is 8 bits, so 1/255 is its floor
and 1/256 is inexpressible.

There is no enable bit in the register block. The reset value encodes "off".
Almost certainly `handshakes_per_window = 0` means unlimited, but if it means
zero handshakes permitted then writing the window first stalls the NoC dead.
That is why the class **reads the live values, writes, and puts them back**
rather than assuming 0. Print the saved values on apply (`TTNOP_VERBOSE` is
enough) so the first simulator run can confirm what "off" actually was.

`noc_set_cfg_reg` in `tt_metal/hw/firmware/src/tt-2xx/quasar/noc.c` is the
kernel-side equivalent. Do not use it. This needs a host poke with no
rebuild.

## Drop-in module

`noc_throttle.py` next to this file is the class. Copy it into the ttnop
directory (it can stay named `noc_throttle.py`) and import `NocThrottle`
from it. Do not paste the class into `ttnop_plugin.py`.

`window == 0` is a no-op: no reads, no writes. That is how an unthrottled
sweep stays cheap.

## Four edit sites, nothing else

### 1. Add the setting to `sweep.py`

Add a field:

```python
# Throttle this tile's NoC injection to one handshake per this many cycles for
# the whole sweep. 0 leaves the NoC alone. Quasar only.
noc_window: int = 0
```

Read it in `from_env`:

```python
noc_window=int(os.environ.get("TTNOP_NOC_THROTTLE", "0") or 0),
```

Validate after the filler check, before the empty-delays check:

```python
if config.noc_window:
    # A window of 1 is one handshake per cycle, which is no throttle at all,
    # and the registers only exist on Quasar.
    if config.noc_window < 2:
        raise ValueError(
            f"TTNOP_NOC_THROTTLE={config.noc_window} is not a throttle. "
            "the window is in cycles, so 256 means 1/256"
        )
    if config.arch != "quasar":
        raise ValueError(
            f"TTNOP_NOC_THROTTLE needs the NoC throttler, which {config.arch} does not have"
        )
```

Do not add a new variant axis. The throttle is a property of the whole sweep,
not of one delay.

### 2. Apply it around the sweep in `ttnop_plugin.py`

Import:

```python
from noc_throttle import NocThrottle
```

Add a helper next to `_injector_for_device`:

```python
def _noc_throttle(self) -> NocThrottle:
    location = TestConfig.TENSIX_LOCATION
    return NocThrottle(
        self.config.noc_window,
        read_word=lambda addr: read_word_from_device(location, addr),
        write_word=lambda addr, word: write_words_to_device(location, addr, [word]),
        log=emit if self.verbose else None,
    )
```

In `Perturber.sweep`, wrap **both** `_prove_reproducible` and `sweep_module.run`
in the context manager. Today those two sit like this:

```python
self._item = item
self._kwargs = _test_kwargs(item)
self._prove_reproducible(item)
try:
    return sweep_module.run(...)
finally:
    ...
```

Change that to:

```python
self._item = item
self._kwargs = _test_kwargs(item)
try:
    # The reproducibility re-run belongs inside the throttle: it is the run
    # every variant is compared against, so it has to see the same NoC.
    with self._noc_throttle():
        self._prove_reproducible(item)
        return sweep_module.run(
            self.config,
            variants,
            self,
            lambda variant, fails, tags, error: self._record(
                item, variant, fails, tags, error
            ),
        )
finally:
    ...
```

This placement matters:

- An ELF reload does not touch NoC config, so applying once per sweep (not
  per variant) is enough and is the cheap choice.
- `_prove_reproducible` is the run every variant is compared against. If it
  sees an unthrottled NoC and the variants see a throttled one, every variant
  reads as drift and the throttle is charged to the delay.
- The clean pytest baseline (the first body run, before `sweep()`) stays
  unthrottled. That is intentional: a red baseline is skipped, not swept, and
  the throttle must not turn a passing test into a skip. Drift is measured
  against the in-sweep re-run, which is throttled.

`write_words_to_device` takes a list. The helper wraps a single word as `[word]`.

Pass `log=emit` only when verbose, matching how the plugin already prints
each detour (`emit` writes to `/dev/tty` so focus.sh can show it live).

### 3. Add it to the report environment

Add an optional argument so existing callers still work:

```python
def environment(
    arch: str, site_mode: str, filler: str, drift: bool = True, noc_window: int = 0
) -> dict:
```

And one dict entry:

```python
"noc": f"throttled to 1/{noc_window}" if noc_window else "unthrottled",
```

Then in `ttnop_plugin.py` `pytest_sessionfinish`, pass `config.noc_window` as
the fifth positional (or as a keyword). A throttled run must be identifiable
in `report.md` or two sweeps will be compared as if they were the same
experiment.

### 4. Document the setting in `README.md`

Table row:

```
| `TTNOP_NOC_THROTTLE` | NoC window in cycles, default 0 (off). `256` is 1/256. Quasar only |
```

Section, after the env table, before Drift:

```
## Throttling the NoC

A second setting, independent of the fillers. A filler moves one thread against
another, `TTNOP_NOC_THROTTLE` moves every NoC transfer against all of them.
`TTNOP_NOC_THROTTLE=256` holds this tile's NIU to one handshake per 256 cycles.
`THROTTLER_CYCLES_PER_WINDOW=256` and `THROTTLER_HANDSHAKES_PER_WINDOW_NIU=1`,
poked over `NOC_CFG` from the host, so no kernel changes and no rebuild.

It is applied once around the whole sweep rather than per variant. An ELF reload
does not touch NoC config, and the reproducibility re-run every variant is compared
against has to see the same NoC. Otherwise the throttle would be charged to the
delay. The reset values are read back and put back, and `TTNOP_VERBOSE` prints
them, because nothing in the register block is an enable and "0 handshakes per
window" could as easily mean a dead NoC as an unthrottled one.

This only works on Quasar. Wormhole and Blackhole have no throttler registers,
so asking for one there is an error. One shared window feeds every per-port
limit for NIU, north, east, south, and west. Only the NIU limit is programmed
because it controls this tile's injection rate. The per-VC throttler
cannot express 1/256 at all. Its `CYCLES_PER_WINDOW` field is 8 bits, so 1/255 is
its floor.
```

## What not to do

- Do not write N/E/S/W unless asked. NIU only.
- Do not use the per-VC throttler. It cannot do 1/256.
- Do not poke from the kernel (`noc_set_cfg_reg`). Host only.
- Do not apply per variant. Once per sweep.
- Do not leave `_prove_reproducible` outside the context manager.
- Do not throttle the clean pytest baseline (the run before `sweep()`).
- Do not assume reset is 0. Read, write, restore.
- Do not accept `TTNOP_NOC_THROTTLE=1`. That is not a throttle.
- Do not silently ignore the env var on WH/BH.
- Do not add a filler, a site, or a delay. This is not that axis.

## How to run it

Quasar simulator, same as any other ttnop sweep, plus the env var:

```
TTNOP_NOC_THROTTLE=256 TTNOP_VERBOSE=1 CHIP_ARCH=quasar pytest ...
```

1/256 on VCS costs what it says. The first run should print
`>> NoC throttled to 1/256 (was [X, Y])`. `X` and `Y` are the live reset
values. Record them. If the device wedges on apply, `Y` was probably "off
means zero handshakes" and this design has to be revisited (write handshakes
before cycles, or treat 0 as a skip).

## Files that must stay untouched

`cave.py`, `scan.cpp`, `tensix_isa.h`, `scanner.py`. The throttle never
enters the cave and has nothing to do with `pacr`.
