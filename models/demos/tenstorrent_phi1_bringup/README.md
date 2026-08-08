# Tenstorrent Bounty #18287: Bring up microsoft/phi-1 on Wormhole (N150/N300)

## Overview
This directory (`tenstorrent_phi1_bringup`) contains our offline modular preparation for Stage 1 bring-up of `microsoft/phi-1` using Tenstorrent's `ttnn` (Tenstorrent Neural Network API) and `tt_transformers` base classes.

## Bounty Status & Architecture
- **Target Issue:** [tenstorrent/tt-metal #18287](https://github.com/tenstorrent/tt-metal/issues/18287)
- **Reward:** $2,500
- **Cloud Hardware Allocation:** Koyeb 1x N300 Wormhole Instance (Activated via promo code `TTDEPLOY25FADEV1M`)
- **Stage 1 Goal:** End-to-end inference (`demo/demo.py`) generating coherent text using `microsoft/phi-1` weights on `N150`/`N300` hardware.

## Directory Layout
```text
tenstorrent_phi1_bringup/
├── README.md               # This document
├── tt/
│   ├── __init__.py
│   └── phi1_model.py       # All Phi-1 components: Attention, MLP, DecoderLayer, Model, CausalLM
├── demo/
│   └── demo.py             # Loader and inference script for Koyeb N300 node
└── scripts/
    └── setup_koyeb_node.sh # Quick setup script when booting up the stateless N300 Koyeb instance
```

## Stateless Cloud Execution Strategy
Because Koyeb N300 instances are **stateless** (data is lost on restart):
1. All code development and architectural refinement occurs here locally (`adraca-laptop` / `adraca-pve`).
2. When running hardware validation, we clone/scp this directory onto our active Koyeb N300 node, execute `demo/demo.py`, record output benchmarks, and pause the node.
