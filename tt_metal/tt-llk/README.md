<div align="center">
<h1>

[Bounties](https://github.com/tenstorrent/tt-llk/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty) | [Buy](https://tenstorrent.com/hardware/blackhole) | [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://job-boards.greenhouse.io/tenstorrent)

</h1>

<img src="./docs/common/_static/tt_llk_refresh_llk_logo.png" alt="llk logo" height="180"/>

<br><br>
**TT-LLK** is Tenstorrent's Low Level Kernel library.

[![STATUS](https://img.shields.io/badge/status-frozen-red)](#)
[![MOVED TO](https://img.shields.io/badge/moved%20to-tt--metal-blue)](https://github.com/tenstorrent/tt-metal)

[![C++](https://img.shields.io/badge/C++-17-green.svg)](#)
[![Python](https://img.shields.io/badge/python-3.10-green.svg)](#)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-llk)
</div>

---

> 🚧 **Repository Status Update**
>
> This repository is now **frozen** and no longer accepts new pull requests.
>
> All ongoing development of **TT-LLK** has moved to the [`tt-metal`](https://github.com/tenstorrent/tt-metal) repository.
>
> 📘 [**Migration guide**](https://github.com/tenstorrent/tt-llk/blob/main/docs/migration_guide.md) - for existing branches and active pull requests
>
> 📁 The LLK source code is now located at:
> ```
> tt-metal/tt_metal/tt-llk
> ```
>
> 🔒 The `main` branch in this repository is **locked**.
>
> 👉 **To contribute, please submit all new pull requests to `tt-metal`.**
>
> This repository will remain available for historical reference.

---

## Overview ##

This repository contains header-only low-level kernels (LLK) for Tenstorrent AI chips, including Wormhole, and Blackhole.

These kernels serve as foundational compute primitives, acting as building blocks for higher-level software stacks that implement machine learning (ML) operations.

Additionally, the repository includes a test environment designed to validate LLK APIs.

## Install ##

1. **Clone the repository**

    Clone this repository to your local computer.

2. **Set up the test environment**

    Follow the instructions in the [testing README](https://github.com/tenstorrent/tt-llk/blob/main/tests/README.md) to set up the test environment. This will also automatically configure pre-commit hooks for code quality checks.

## Software dependencies ##

Test environment requires SFPI compiler for building, which is automatically ingested from <https://github.com/tenstorrent/sfpi>

## Documentation ##

The following documentation is available to help you understand and use low-level kernels:

1. **[Intro](docs/llk/l1/intro.md)**
   A concise introduction to LLKs, designed for both technical and non-technical audiences. This document outlines the scope of the LLK software stack and its relationship to other Tenstorrent software components.

2. **[Top-level Overview](docs/llk/l2/top_level_overview.md)**
   Provides a high-level look at the Tensix Core and Tensix Engine architecture, including data organization for efficient LLK usage and operations supported by LLKs. This document is not tied to any specific chip generation (such as Wormhole) and is aimed at engineers and technical readers who want to understand the general architecture and capabilities.

3. **[LLK Programming Model](docs/llk/l3/programming_model.md)**
   This document dives into architectural details to best explain the usage of the LLK API. It is intended for op writers and advanced users, and connects LLK concepts with our runtime stack, [tt-metal](https://github.com/tenstorrent/tt-metal), providing practical guidance on how to leverage LLKs for efficient kernel development.

## Contributing ##

⚠️ **Note:** This repository is **frozen** and does not accept new pull requests.

We welcome contributions to TT-LLK in its new home! Please contribute via the [`tt-metal`](https://github.com/tenstorrent/tt-metal) repository instead.

If you are looking for historical contribution guidelines, you can still refer to the documents below:

1. **Read the Guidelines**

    Familiarize yourself with our [CONTRIBUTING](https://github.com/tenstorrent/tt-llk/blob/main/CONTRIBUTING.md) guide and [CODE OF CONDUCT](https://github.com/tenstorrent/tt-llk/blob/main/CODE_OF_CONDUCT.md).

## Tenstorrent Bounty Program Terms and Conditions ##

This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-llk, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both 'bounty' and difficulty level labels!

### Note ###

Old LLK repositories:

- <https://github.com/tenstorrent/tt-llk-gs>
- <https://github.com/tenstorrent/tt-llk-wh-b0>
- <https://github.com/tenstorrent/tt-llk-bh>

have been archived. This repository remains available for historical reference, and all ongoing development has moved to [`tt-metal`](https://github.com/tenstorrent/tt-metal).
