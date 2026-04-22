# TT Ethernet tool

`tt-ethtool` is a low-level utility for interacting with the TT fabric ethernet
ports.

## Goals

* Facilitate debug and operations of TT fabric.
* Minimal dependencies (UMD should suffice).
* Do one thing, but do it well.

## Non-goals

* Global control of TT fabric on a distributed system.

## Building

To build `tt-ethtool`:

    cmake --build build --target tt-ethtool
