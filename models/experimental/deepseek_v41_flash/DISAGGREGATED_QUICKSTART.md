# DeepSeek-V4.1-Flash, disaggregated: set up, build, run

This guide runs prefill and decode on two separate Blackhole Galaxies (8x4, 32 chips each), each with its own
host. For one host and one Galaxy, use `QUICKSTART.md` instead.

* The **prefill server** (`demo/prefill_server.py`) runs on one host.
* The **decode server** (`demo/server.py`) runs on the other. It accepts client requests, sends each prompt to
  the prefill server, and loads the KV cache that comes back.

The KV cache can move between the two in one of two ways. You will pick one in section 3.

| Mode | How the KV cache moves | Requires | Time to first token at 1M tokens |
| --- | --- | --- | --- |
| **Fabric push** (faster) | Prefill writes it straight into decode's device memory over TT-Fabric | The Galaxies cabled to each other, and the link test in section 3 passing | 26.4 s |
| **TCP** (fallback) | Prefill copies it to its host and sends it over the network; decode writes it onto its device | A network route between the hosts | 33.6 s |

## Before you start

* **Disk:** ~480 GB on both hosts for the checkpoint, plus ~700 GB on the decode host for the expert weight cache.
* **RAM:** ~210 GB free on the prefill host.
* **Shared cards:** other users may be using the Galaxies. Before any command that opens a device, check:
  `ps -eo user,pid,args | grep -E "tt-run|server.py|pytest"`.

## 1. Set the host variables

Set these in every shell you run commands from, on both hosts:

```bash
export PREFILL_IP=<prefill host's IP>       # decode connects here
export DECODE_IP=<decode host's IP>         # clients connect here
export PREFILL_PORT=9041 DECODE_PORT=8000
```

For fabric push, the prefill host starts both servers, reaching the decode host over ssh at `$DECODE_IP`. Check
that on the prefill host, `ssh -o BatchMode=yes $DECODE_IP hostname` prints the decode hostname without asking
for a password.

The launch script works out the rest from `$DECODE_IP`: the decode hostname, and the network interface MPI runs
over (the one each host routes to the other through). To force an interface, also set
`MPI_IFACE=<name>`, or `MPI_IFACE=<name>,<name>` if the two hosts name it differently.

## 2. Set up both hosts

Do this on each host. Clone to the same path on both hosts, because the fabric push launch starts the decode server
at the path it was run from on the prefill host. The commands below use `~/tt-metal`.

```bash
git clone --branch ds_v4.1_flash --recurse-submodules https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal
sudo ./install_dependencies.sh     # first time only
./build_metal.sh                   # rerun after C++ changes; kernels/ files compile at runtime
./create_venv.sh                   # first time only
source python_env/bin/activate
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017
```

The checkpoint lands in the default Hugging Face cache, where both servers look for it.

**On the decode host only**, two more steps:

1. Build the Rust host runtime. Without it, decode runs slower and has no web monitor. If `cargo` is missing,
   first run `curl https://sh.rustup.rs -sSf | sh -s -- -y && source ~/.cargo/env`.

   ```bash
   models/experimental/deepseek_v41_flash/host_rs/build.sh
   ```

2. Fill the expert weight cache. This runs on the host only, without the device. If you skip it, the first
   server start converts the experts one at a time.

   ```bash
   python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --all-layers
   python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --dspark --dtype bfloat8_b
   ```

## 3. Start the servers: pick one option

Use **option A (fabric push)** if the Galaxies are cabled to each other. If its link test fails, or you need to
start and stop the two servers separately, use **option B (TCP)**.

Run all commands from the clone (`~/tt-metal`) with the venv active.

### Option A: fabric push

#### Step 1: test the link

Run this before the first launch, and again after re-cabling, after `tt-smi -glx_reset`, or after a failed push.
If either test fails or hangs, use option B.

On the prefill host, set up the MPI arguments for the tests. `IFACE` is the interface the prefill host routes to
the decode host through. If the decode host names its side differently, append that name with a comma, as in the
`MPI interface:` line the launch script prints.

```bash
CFG=tests/tt_metal/distributed/config
DECODE_HOST=$(ssh $DECODE_IP hostname)
IFACE=$(ip -o route get $DECODE_IP | sed -n 's/.* dev \([^ ]*\).*/\1/p')
printf 'rank 0=%s slot=0\nrank 1=%s slot=0\n' $(hostname) $DECODE_HOST > /tmp/d_p.rankfile
MPI="--host $(hostname),$DECODE_HOST --map-by rankfile:file=/tmp/d_p.rankfile \
  --mca btl self,tcp --mca btl_tcp_if_include $IFACE --bind-to none --tag-output"
```

Rank 0 is the prefill host and rank 1 is the decode host. `tt-run` prefixes each output line with its rank, as
`[1,0]` or `[1,1]`. The rank-binding files have `40_41` in their names after the machines they were written
for. They work on any pair of hosts, because the rankfile above picks the hosts.

**Test 1: traffic across the cables.** Joins both Galaxies into one 16x4 mesh and runs an `all_reduce` across
the host boundary.

```bash
tt-run --rank-binding $CFG/dual_8x4_hosts_40_41_rank_bindings.yaml --mpi-args "$MPI" \
  bash -c 'cd $TT_METAL_HOME && exec python tests/tt_metal/multihost/test_fabric_cross_host.py'
```

Pass: both ranks print `ALL CHECKS PASSED`.

**Test 2: the KV write the servers do.** Rank 0 writes into KV-cache-shaped buffers on rank 1's device over the
fabric, using the servers' packet size. Keep `--payload 6208`.

```bash
tt-run --rank-binding $CFG/dual_bh_galaxy_8x4_hosts_40_41_two_mesh_rank_bindings.yaml --mpi-args "$MPI" \
  bash -c "cd \$TT_METAL_HOME && D=models/experimental/deepseek_v41_flash/tests/xhost_kv
    if [ \"\$OMPI_COMM_WORLD_RANK\" = 0 ]; then exec python \$D/sender_write.py --payload 6208; fi
    exec python \$D/receiver_init.py --sender $PREFILL_IP --payload 6208"
```

Pass: rank 1 prints `PASS`. Then, after the job exits, run this on the decode host to confirm the bytes are
still in device memory. It also prints `PASS`, and does not open the device:

```bash
python models/experimental/deepseek_v41_flash/tests/xhost_kv/receiver_verify.py
```

#### Step 2: launch

On the prefill host:

```bash
models/experimental/deepseek_v41_flash/demo/run_d_p_servers.sh --prefill-port $PREFILL_PORT -- \
  --host 0.0.0.0 --port $DECODE_PORT --prefill-url $PREFILL_IP:$PREFILL_PORT --kv-transfer-mode fabric
```

This one command starts both servers as a single `tt-run` job: prefill on this host, decode on the host at
`$DECODE_IP` (which must be exported). Ctrl-C here stops both. Before launching, it prints what it worked out:

```
prefill: <prefill hostname>, decode: <decode hostname> (<DECODE_IP>), MPI interface: <interface>
```

A cold start takes about 6.5 minutes: ~200 s loading weights, ~160 s compiling, ~22 s warming up. Watch for these
log lines, in order:

1. `prefill server on ...`: prefill is listening.
2. `disaggregated prefill connected to ... KV pushed over TT-Fabric`: decode has connected to prefill.
3. `serving 'deepseek-v4.1-flash' on http://...`: ready for requests.

### Option B: TCP

You start the two servers separately, one on each host. Either one can be restarted on its own. Decode
reconnects on the next prompt.

On the prefill host:

```bash
python models/experimental/deepseek_v41_flash/demo/prefill_server.py --port $PREFILL_PORT
```

On the decode host:

```bash
python models/experimental/deepseek_v41_flash/demo/server.py --host 0.0.0.0 --port $DECODE_PORT \
  --prefill-url $PREFILL_IP:$PREFILL_PORT --kv-transfer-mode tcp --prefill-wait 7200
```

Decode retries the prefill server for up to `--prefill-wait` seconds (two hours) while prefill loads. Decode is
ready when it logs `serving 'deepseek-v4.1-flash' on http://...`.

To test the TCP path on its own, run this on the decode host once prefill logs `prefill server on ...`. It needs
no device. Pass: it prints `PASS`.

```bash
python models/experimental/deepseek_v41_flash/demo/prefill_client.py --server $PREFILL_IP:$PREFILL_PORT --random 16384 --repeat 3
```

## 4. Send a request

From any machine that can reach the decode host, with `DECODE_IP` and `DECODE_PORT` set:

```bash
curl -s --no-buffer -X POST http://$DECODE_IP:$DECODE_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

The web monitor at `http://$DECODE_IP:8090` shows prefill progress, decode tokens/s, and KV cache usage. It needs
the Rust runtime from section 2. The page stays up even after the decode server exits, so use the curl above to
check whether the server is running.

To measure performance across prompt lengths:

```bash
python models/experimental/deepseek_v41_flash/demo/perf_client.py --url http://$DECODE_IP:$DECODE_PORT --min-len 32k --decode-tokens 3072
```

Reference numbers from one run of each mode, with one user and DSpark on (the default; `--no-dspark` turns it
off).

Fabric push:

| Prompt length | Time to first token (s) | Prefill tokens/s | Decode tokens/s |
| --- | --- | --- | --- |
| 32K | 1.40 | 23,470 | 189.5 |
| 128K | 3.48 | 37,668 | 183.7 |
| 512K | 12.09 | 43,365 | 165.0 |
| 1M | 26.44 | 39,542 | 153.0 |

TCP:

| Prompt length | Time to first token (s) | Prefill tokens/s | Decode tokens/s |
| --- | --- | --- | --- |
| 32K | 3.37 | 9,718 | 164.6 |
| 64K | 4.29 | 15,286 | 170.6 |
| 128K | 5.47 | 23,961 | 133.5 |
| 256K | 7.65 | 34,278 | 146.0 |
| 512K | 15.11 | 34,700 | 132.8 |
| 1M | 33.59 | 31,124 | 139.2 |

## 5. Troubleshooting

| What you see | What to do |
| --- | --- |
| `set DECODE_IP to the decode host's IP` | Export `DECODE_IP` in the shell you launch from (section 1). |
| `ssh to DECODE_IP=... failed`, or `tt-run` cannot reach the decode host | Set up passwordless ssh from the prefill host to `$DECODE_IP` (section 1). |
| `no interface between ...`, or the job hangs before any rank prints | Find the interface the hosts reach each other on (`ip -br addr`) and set `MPI_IFACE` (section 1). |
| Fabric init times out, or a rank reports links down | Rerun test 2 in section 3, option A. If it still fails, use option B. |
| `tt-smi` shows fewer than 32 chips after a hang | Run `tt-smi -glx_reset` on that host, then rerun the link test. |
| Import error, or an op exists on one host but not the other | The clones are on different commits, or one host was not rebuilt (section 2). |
| Decode rank fails with `No such file or directory` at launch | The decode host's clone is not at the same path as the prefill host's (section 2). |
| `prefill server ... not listening yet; retrying` | Prefill is still loading. Decode gives up after `--prefill-wait` seconds. |
| Decode cannot connect to the prefill server | From the decode host, run `nc -zv $PREFILL_IP $PREFILL_PORT`. If it fails, fix the IP or the network route. |
| The first prefill start spends many minutes building Engram tables | Expected. They are saved to `/dev/shm/deepseek_v41_d_p_engram`, and later starts reuse them. |

More detail: `PREFILL_D_P_FABRIC_PUSH.md` (fabric push), `PREFILL_D_P_INTEGRATION.md` (TCP),
`PREFILL_D_P_PREFIX_CACHE.md` (prefix cache).
