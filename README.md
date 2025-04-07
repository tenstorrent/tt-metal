# tt-llk: CPP Low Level Kernels (LLK) & test infrastructure #

## Overview ##

This repository contains header-only low level kernels for Tenstorrent AI chips (Grayskull (deprecated), Wormhole_B0, and Blackhole), which represent foundational primitives of compute used as building blocks for higher level software stacks that implement ML OPs. Alongside the kernels is a test environment used for validating LLK APIs.

## Install ##

1. Clone repository
2. Set up test environment per testing [README](https://github.com/tenstorrent/tt-llk/blob/main/tests/README.md)

## Software dependencies ##

Test environment requires SFPI compiler for building, which is automatically ingested from <https://github.com/tenstorrent/sfpi>

## Contributing ##

1. Go over [CONTRIBUTING](https://github.com/tenstorrent/tt-llk/blob/main/CONTRIBUTING.md) guide
2. Create a new branch.
3. Make your changes and commit.
4. Add new tests to cover your changes if needed and run existing ones.
5. Start a pull request (PR).

## Tenstorrent Bounty Program Terms and Conditions ##

This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-llk, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both 'bounty' and difficulty level labels!

### Note ###

Old LLK repositories:

- <https://github.com/tenstorrent/tt-llk-gs>
- <https://github.com/tenstorrent/tt-llk-wh-b0>
- <https://github.com/tenstorrent/tt-llk-bh>

have been archived. All ongoing development continues in this repository.
