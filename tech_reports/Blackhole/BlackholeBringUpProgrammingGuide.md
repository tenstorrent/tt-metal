# Blackhole Bring-Up Programming Guide

## Introduction

Information relevant to programming Blackhole while it is being brought up.

## Wormhole N150 vs. Blackhole

<table><thead>
  <tr>
    <th rowspan="3"></th>
    <th colspan="3">Tensix</th>
    <th colspan="2">Ethernet</th>
    <th colspan="3">DRAM</th>
    <th colspan="4">NoC</th>
  </tr>
  <tr>
    <th rowspan="2">Total</th>
    <th rowspan="2">Available for Compute</th>
    <th rowspan="2">L1</th>
    <th rowspan="2">Total</th>
    <th rowspan="2">Programmability&nbsp;&nbsp;</th>
    <th rowspan="2">Total</th>
    <th rowspan="2">Bank Size </th>
    <th rowspan="2">Programmability</th>
    <th colspan="3">Alignments</th>
    <th rowspan="2">Multicast</th>
  </tr>
  <tr>
    <th>DRAM</th>
    <th>PCIe</th>
    <th>L1</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Wormhole N150</td>
    <td>8x10</td>
    <td>8x8</td>
    <td>1464 KB</td>
    <td>16</td>
    <td>1x RISC-V<br>256 KB L1</td>
    <td>12 banks</td>
    <td>1 GB</td>
    <td>N/A</td>
    <td>Read: 32B<br>Write: 16B</td>
    <td>Read: 32B<br>Write: 16B</td>
    <td>Read: 16B<br>Write: 16B</td>
    <td>Rectangular</td>
  </tr>
  <tr>
    <td>Blackhole</td>
    <td>14x10</td>
    <td>13x10</td>
    <td>1464 KB<br>Data cache added </td>
    <td>14</td>
    <td>2x RISC-V<br>512 KB L1</td>
    <td>8 banks</td>
    <td>~4 GB</td>
    <td>1x RISC-V<br>128 KB L1</td>
    <td>Read: 64B<br>Write: 16B</td>
    <td>Read: 64B<br>Write 16B</td>
    <td>Read: 16B<br>Write: 16B</td>
    <td>Rectangular<br>Strided<br>L-shaped</td>
  </tr>
</tbody></table>

### L1 Data Cache

Blackhole added a data cache in L1. Writing an address on one core and reading it from another only requires the reader to invalidate if the address was previously read.

Invalidating the cache can be done via calls to `invalidate_l1_cache()`

The cache can be disabled through an env var:
```
export TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS=<BR,NC,TR,ER>
```

### Ethernet Cores

Runtime has not enabled access to second RISC-V on the ethernet cores yet.

Fast dispatch can be run out of ethernet cores.

### DRAM

Runtime has not enabled access to program RISC-V on DRAM yet.

### NoC

Non-rectangular multicast shapes have not been tested yet.

BH enabled 16-deep FIFOs for each of the four command buffers. These are enabled by default in `noc_init` as BH cmd\_buffer has known issues. NoC APIs are not impacted by this change.

## Debug

Debug tools are functional on BH and it is reccomended to use Watcher when triaging Op failures to catch potential alignment issues. Disabling the L1 cache can be helpful to identify missed cache invalidations.

## Resetting

Depending on the firmware, reset via `tt-smi -r 0` may not work and the board will need to be rebooted.

## CI

Bringing up full post commit is a WIP on BH, currently we only run the cpp tests. It is triggered on pushes to main but we have seen some instability with the machines with ND failures.

## Issue Tracking

Please file issues or any instances of ND behaviour to the Blackhole [board](https://github.com/orgs/tenstorrent/projects/50/views/1)
