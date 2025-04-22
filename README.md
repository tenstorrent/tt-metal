<div align="center">
<h1>

[Bounties](https://github.com/tenstorrent/tt-llk/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty) | [Buy](https://tenstorrent.com/cards/) | [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://job-boards.greenhouse.io/tenstorrent)

</h1>

<img src="./docs/common/_static/tt_llk_refresh_llk_logo.png" alt="llk logo" height="180"/>

<br><br>
**TT-LLK** is Tenstorrent's Low Level Kernel library.

[![C++](https://img.shields.io/badge/C++-20-green.svg)](#)
[![Python](https://img.shields.io/badge/python-3.8%20|%203.10-green.svg)](#)
</div>

## Overview ##

This repository contains header-only low-level kernels for Tenstorrent AI chips, including Grayskull (deprecated), Wormhole, and Blackhole.

These kernels serve as foundational compute primitives, acting as building blocks for higher-level software stacks that implement machine learning (ML) operations.

Additionally, the repository includes a test environment designed to validate LLK APIs.

## Install ##

1. **Clone the repository**

    Clone this repository to your local computer.

2. **Set up the test environment**

    Follow the instructions in the [testing README](https://github.com/tenstorrent/tt-llk/blob/main/tests/README.md) to set up the test environment.

## Software dependencies ##

Test environment requires SFPI compiler for building, which is automatically ingested from <https://github.com/tenstorrent/sfpi>

## Contributing ##

We welcome contributions to improve tt-llk! Please follow these steps to get started:

1. **Read the Guidelines**

    Familiarize yourself with our [CONTRIBUTING](https://github.com/tenstorrent/tt-llk/blob/main/CONTRIBUTING.md) guide and [CODE OF CONDUCT](https://github.com/tenstorrent/tt-llk/blob/main/CODE_OF_CONDUCT.md).

2. **Create a Branch**

    Create a new branch for your changes.

3. **Make Changes**

    Implement your changes and commit them with clear and descriptive messages.

4. **Add Tests**

    If applicable, add new tests to cover your changes and ensure all existing tests pass.

5. **Submit a Pull Request**

    Open a pull request (PR) to propose your changes for review.

## Tenstorrent Bounty Program Terms and Conditions ##

This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-llk, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both 'bounty' and difficulty level labels!

### Note ###

Old LLK repositories:

- <https://github.com/tenstorrent/tt-llk-gs>
- <https://github.com/tenstorrent/tt-llk-wh-b0>
- <https://github.com/tenstorrent/tt-llk-bh>

have been archived. All ongoing development continues in this repository.
