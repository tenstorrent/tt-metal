# Capture and UMD

Capture retrieves router state from chips through UMD. This doc covers how Capture uses UMD and the UMD behavior
that explains the choices in `fabric_capture_umd.cpp`, the only file with UMD code. For what capture does with the
bytes, see the capture [README](../README.md).

## How capture uses UMD

`UmdDeviceReader` owns a `tt::umd::Cluster` of its own. It shares nothing with a workload's cluster, so it works on a
run that has exited, been killed, or is stuck. Below is a summary of some key choices in terms of interacting with UMD:

| Choice | Why |
| --- | --- |
| `num_host_mem_ch_per_mmio_device = 0` | Maps no hugepages. See "No hugepages" |
| `eth_fw_heartbeat_failure = IGNORE` | Discovery skips a core whose heartbeat has stalled instead of throwing. See "Discovery and stalled heartbeats" |
| `cluster_id` from `TT_METAL_CLUSTER_ID` | Names the host as the run named it. Null leaves UMD's default, the host name |
| Never `start_device` or `close_device` | The cluster resets nothing and writes nothing to a chip. UMD closes only what `start_device` opened, so the destructor closes nothing |
| `has_authentic_chip_unique_ids()` must hold | Capture finds the manifest's chips by `asic_id`. If UMD didn't read the ids from the chips, the constructor fails |

### Reads

- `read_l1` calls `tt::umd::Cluster::read_from_device`, for L1.
- `read_register` calls `tt::umd::Cluster::read_from_device_reg`, for registers.
- Both address an Ethernet core in translated coordinates, `CoordSystem::TRANSLATED`. 
- For information on remote reads, look at 

## Reorg to phrase around the purppose of collect umd cluster facts
### What `collect_umd_cluster_facts` takes

From the cluster's descriptor and SoC descriptors:

- the cluster id;
- for each chip UMD opened: its chip id, its unique id, its architecture, and the translated core of each Ethernet
channel;
- the Ethernet cores discovery skipped. Discovery records a stalled heartbeat as a health erro (`EthFirmwareHeartbeatError`) at the core it checked. Capture translates that core to translated coordinates and keeps
UMD's message.

## Remote chip interactions

On Wormhole, only the chips on PCIe are MMIO. A T3K has 4 of them, and the other 4 are remote (non-MMIO), so UMD reads
them through an MMIO chip, aka. the gateway, by sending requests over Ethernet. The gateway's Ethernet core takes the
request into its L1 and base firmware on that core services it. This path is the source of most quirks below.

On Blackhole every galaxy chip is MMIO. The remote-read quirks don't apply, and capture's choice of a gateway channel
does nothing.

## Quirks and what capture does about them

### No hugepages

With hugepages mapped (UMD's default), UMD returns a remote read over 1 KB (`256 * DATA_WORD_SIZE`) through host DRAM,
in blocks of `ETH_ROUTING_BLOCK_SIZE` (32 KB), instead of through the gateway Ethernet core's buffer
(`remote_communication_legacy_firmware.cpp`).

On a T3K with routers running, every other such read came back corrupted. A 2704-byte routing table read alternated
between the right bytes and bytes like `0xffb01737`, in lockstep across all cores and chips. Four-byte reads of the same
word were always right, and with no hugepages every large read was right. Reads of 1 KB or less never take the host
DRAM path.

The cause isn't known. One guess is that the workload process uses the same hugepage response buffers as capture's
cluster. That is untested.

With `num_host_mem_ch_per_mmio_device = 0`, a read over 1 KB from a remote chip goes in blocks of `MAX_BLOCK_SIZE`
(1 KB) through the gateway Ethernet core's L1. It's slower. UMD logs once per remote chip:

```
Large transfer to remote chip without system memory setup. Performance will be degraded.
```

That warning is about speed only. A read still returns the right bytes.

### Discovery and stalled heartbeats

Discovery waits up to 50 ms (`ETH_HEARTBEAT_TIMEOUT`) for each Ethernet core's heartbeat to advance. On Wormhole that
heartbeat is base firmware's (signature `0xABCD`), and base firmware runs only when the fabric router hands it the
core. A router busy with traffic rarely does. With UMD's default options, discovery of a T3K running a workload threw
`EthFirmwareHeartbeatError` (`Stuck at 0xabcdd4c1`) on every attempt, before any read.

With `IGNORE`, discovery logs the core and doesn't discover through it. On that T3K it skipped 16 cores and still found
all 8 chips, and every read worked.

A remote chip reachable only through skipped cores would go undiscovered, so capture would not find it. None was missed
in that run.

Blackhole's heartbeat isn't checked during discovery, so a Blackhole chip never reports a skipped core.

### Remote reads and the gateway's router

A remote read is served by base firmware on the gateway's Ethernet core, so its speed depends on whether that core's
router hands the core over. UMD reads through the first core in the gateway's active-channel list, whatever that
core's router is doing, by default. It moves on to the next core only in its write path (`write_to_non_mmio`), when a
core's command queue fills; reads never do. This can result in slowness of reads, ultimately bottlenecked by the
router on the gateway core handing over control to base firmware (who is responsible for forwarding the read request) to
the router on the remote chip.

Measured on a T3K, three four-byte reads took:

| Gateway channel | Three reads |
| --- | --- |
| Busy | about 255 to 282 ms |
| Idle | about 270 to 291 us |

So `route_remote_reads_through_fastest_channel` picks the channel for each remote chip, once, when the reader opens:

1. The candidates are the gateway channels linked straight to the remote chip. If there are none (the chip is more than
one hop away), they are every active channel of the gateway.
2. For each candidate, it makes that channel the chip's only remote transfer core (`set_remote_transfer_ethernet_cores`)
and times 3 four-byte reads of a Tensix core's L1 at address 0.
3. It reads through the fastest, and prints each candidate's time and its choice to stderr. A candidate whose reads
throw is printed as failed and skipped.
4. If no candidate answers, it restores UMD's default channels and says so.

Capturing that T3K's 40 routers took about 3.6 s this way, against about 5.5 minutes through the busy channel.

The limit: if every link to a remote chip is busy, the choice is the least slow of them, which may still be slow.

### Capture's reads can show up in what it reads

On Wormhole, a router's heartbeat word reads as base firmware's (an `0xABCD...` prefix) when the router has handed its
core over. That happens on idle routers, and on routers carrying UMD's remote reads, which includes capture's own.
Decode's activity verdict treats such a read as not the router's.

### Opening the cluster is slow

Opening the cluster takes about 30 s on a T3K. Discovery finished in under a second, then about 30 s passed until UMD
logged "Opening local chip ids". The cause hasn't been investigated.

## How tt-exalens differs

[tt-exalens](https://github.com/tenstorrent/tt-exalens) is another UMD client. From its source:

- [`ttexalens/umd_api.py`](https://github.com/tenstorrent/tt-exalens/blob/main/ttexalens/umd_api.py) uses
`TopologyDiscovery::discover` and raw `TTDevice`s, not a `Cluster`. Its remote devices have no host memory either, so
they take the same 1 KB path.
- The same file sets UMD's log level to Error unless `TT_LOGGER_LEVEL` is set or exalens is verbose, which hides the
large-transfer warning.

Future work may entail using raw `TTDevices` like tt-exalens, or using the tt-exalens c++ backend directly.