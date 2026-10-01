# Repro handoff: lost Counter increment in PR #58242 multicast helpers (Blackhole)

This branch is **PR #58242 (`sjovic/mcast-helpers-conv2d-conv3d`, head `b68c5659687`) plus one repro
test**. Nothing in the library is changed. The goal is to show on real BH hardware whether the race
below actually happens.

Do **not** push, comment on the PR, or open issues. Report results back to the person who handed this over.

## Suspected bug

`ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_pipe.inl:95-97` (`send()`) and `:198-200` (`send_signal()`):

```cpp
if constexpr (ROTATING_SENDER && DATA_READY_SIGNAL == DataReadySignal::Counter) {
    data_ready_.up(1);   // local self-increment of this core's data_ready counter
}
```

In a rotating channel with `DataReadySignal::Counter`, every core's `data_ready` word is a monotonic
counter, and receivers wait with `wait_min(round + 1)`. The round's sender isn't included in its own
atomic multicast, so it increments its own counter locally. On WH/BH the semaphore scope is always
`LOCAL_NONATOMIC`, and `Semaphore::up(v)` is a plain RISC `*ptr += v`
(`tt_metal/hw/inc/api/dataflow/semaphore_dm_impl.h:94`): load, add, store.

With `McastConfig{.handshake = false}`, the next round's sender can land its NoC atomic increment
(`inc_multicast`) on that same L1 word between the load and the store. One increment is lost, the
counter stays one short, and that core hangs forever in `wait_min()` at its next receive.

```
     S (round r sender)                     T (round r+1 sender)
t0   inc_multicast -> rectangle
t1                                          atomic lands, T.counter = r+1, ack -> S
t2                                          wait_min(r+1) returns; no handshake, so
                                            T sends immediately (send_signal: no payload)
t3   async_atomic_barrier() returns
t4   up(1): LOAD v
t5                                          T's inc_multicast lands on S: v -> v+1
t6   up(1): STORE v+1   <-- T's increment lost
     ... S: wait_min() never satisfied -> hang
```

With the handshake on, this can't happen: T needs S's ack, which S only sends from its next
`receive()`, after the store. The host (`mcast_host_impl.cpp`) doesn't reject
rotating + Counter + `handshake=false`, and no existing test covers it (every `handshake=False` test uses
a fixed sender, mostly with `rounds=1`).

The race window is only one L1 load plus a store, so expect it to be **rare per round**. The repro runs
many rounds and sweeps the receive-to-send turnaround to raise the odds.

## What the repro does

- `kernels/repro_rotating_counter_race.cpp`: N cores on row 0 (logical (0,0)..(N-1,0)). Every core is
  both a sender (on its phase) and a receiver (on the other phases) of one rotating channel, using
  only `send_signal()` / `receive_signal()` for the fastest turnaround. After each receive it spins
  `0..max_delay` nops (pseudo-random per round) before its next operation.
- **Without a handshake**, each receive is preceded by a bounded poll of the same counter word
  (`timeout_polls`). On a timeout the core records `{round, observed, expected}` and leaves the loop,
  so the program always finishes instead of hanging. Each core writes a 16-word status row to DRAM.
- **With the handshake (control)**, the bounded poll is skipped, because the sender waits for this
  core's ack. The same kernel and topology must pass every time; if the control hangs, the harness is
  broken, not the library.
- `test_repro_rotating_counter_race.py`: parametrized over `handshake` (no_handshake /
  handshake_control), `n_cores` (2, 3, 4, 8), `max_delay` (0, 32, 128, 512) and `noc` (0, 1).
  200k rounds per launch, 5 launches per case.

The kernel and test have **not yet been compiled or run**; they were written without hardware access.
First-run compile or API fixes in the two repro files are expected and fine. Do not change anything
under `ttnn/cpp/ttnn/kernel_lib/` or `tt_metal/`.

## How to run (BH machine)

```bash
git fetch <remote> vvukomanovic/repro-pr58242-rotating-counter-race   # or however the branch reached you
git checkout vvukomanovic/repro-pr58242-rotating-counter-race
git submodule update --init --recursive
./build_metal.sh            # the PR adds host code + nanobind (ttnn.Mcast); a full build is required
source python_env/bin/activate

# The control first: must pass. If it fails or hangs, fix the harness before going further.
pytest -svx "tests/ttnn/unit_tests/kernel_lib/test_repro_rotating_counter_race.py::test_rotating_counter_lost_increment[handshake_control-2-0-0]"

# The suspected-broken configuration
pytest -sv tests/ttnn/unit_tests/kernel_lib/test_repro_rotating_counter_race.py -k no_handshake

# Full sweep including control
pytest -sv tests/ttnn/unit_tests/kernel_lib/test_repro_rotating_counter_race.py
```

Use `pytest --collect-only -q` to see the exact parametrized ids, since `-k` matching depends on them.
Also run the PR's own rotating/Counter tests once as a sanity check that the build is good:

```bash
pytest -sv tests/ttnn/unit_tests/kernel_lib/test_mcast_device_api.py -k "rotating_line_counter_smoke or positional_control"
```

Hang safety: no-handshake cases can't hang (bounded polls); the control cases use the real
unbounded wait. Consider `TT_METAL_WATCHER=10` for the control run if anything stalls.

## Reading the result

- A failing no-handshake case logs lines like
  `launch L: core C round R: counter X < expected X+1 (final F, sent S); K/N cores timed out`.
  The core with the **earliest failing round** is the one that lost the increment. The other cores
  then time out because that core stops sending. `observed == expected - 1` is the signature of
  exactly one lost increment.
- A passing case checks that every core finished all rounds and ends with `final counter == rounds`.

## If it does not reproduce

The window is small, so a pass isn't proof of safety. In this order:
1. Increase `rounds` (e.g. 2M) and `launches` in the test.
2. Use a finer `max_delay` sweep near 0 (e.g. 0..64 step 4). The window is a few tens of cycles,
   and the right offset depends on the NoC distance between S and T.
3. Try non-adjacent senders by spreading cores apart (receivers on x = 0, 3, 6, ...). That needs a
   multi-rectangle receiver set, so only try it if the simple line doesn't reproduce.
4. Confirm the race exists by construction: disassemble the kernel ELF
   (`riscv-tt-elf-objdump -d` on the built kernel in `$TT_METAL_CACHE` or `~/.cache/tt-metal-cache`)
   and check that the self-increment after `send_signal` is a separate `lw` / `addi` / `sw` on the
   semaphore address, not an atomic.

## What to report back

- Build commit (`git rev-parse HEAD`), board (`tt-smi` / BH SKU), and the exact pytest commands.
- Pass/fail counts per parametrization (the final `logger.info` line of each case).
- For failures: the logged first-failure line(s), with core, round, observed and expected.
- Any changes you had to make to the two repro files to get them to compile or run.
