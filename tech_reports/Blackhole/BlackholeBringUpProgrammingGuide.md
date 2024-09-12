# Blackhole Bring-Up Programming Guide

## Introduction

Information relevant to programming Blackhole while it is being brought up.

## Memory Alignment

- 32 bytes for L1
- 64 bytes for DRAM

## Command Buffer

BH cmd\_buffer is known to have issues, need to turn on cmd\_buffer\_fifo.
Instead of using noc\_cmd\_buf\_ready and cmd\_buffer for sending out mcast requests (as well as other read/write requests),
use cmd\_buffer\_fifo and CMD\_BUF\_AVAIL

## tt-smi

Depending on the firmware, tt-smi reset may not work and the board will need to be rebooted.
