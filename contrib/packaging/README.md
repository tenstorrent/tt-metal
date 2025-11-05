# Community-Driven Linux Distribution Support

Welcome to the community-driven Linux distribution support folder for Tenstorrent's tt-metal repository! This folder is dedicated to providing scripts, installers, packaging, and guides for getting Tenstorrent software and hardware working on various Linux distributions beyond what Tenstorrent officially supports. While Tenstorrent does not officially support these distributions, we believe in fostering collaboration and resource-sharing among our users. Thank you to all contributors who help make this possible!

## Community Support List

| Distro | Native Packaging | Scripts | Guides |
| ------ | ---------------- | ------- | ------ |
| Debian 12    | ❌ | ✅ | ❌ |
| QubesOS 4    | ❌ | ❌ | ✅ |
| Arch Linux   | ✅ | — | ✅ |
| Alpine 3.21  |    |    |    |
| LinuxMint 22 |    |    |    |


## Folder Structure
The folder structure is organized as follows:
tt-metal.git/contrib/packaging/[DISTRO]/...

Where `[DISTRO]` is replaced with the name and version of the Linux distribution (e.g., `debian-12`, `fedora-42`, `archlinux` <small>(btw)</small>, etc.). Each distribution-specific folder will contain relevant scripts, installers, and/or guides contributed by the community.

## Contributing
If you would like to contribute to this effort, please review the following guidelines:
0. Read our contribution guide [here](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md)
1. Check our issue tracker for existing work related to your distribution.
2. Create a new folder under `/contrib/packaging/` for your Linux distribution if one does not already exist.
3. Include detailed documentation (e.g., a `README.md` file) within your distribution folder that explains how to use the provided scripts or installers.
4. Ensure that any scripts or installers you provide are well-commented and follow best practices for security and maintainability.
5. Submit your contribution via a pull request for review by the community.

**Note**: By opening a PR, contributors must follow all terms outlined in the Licensing section below.

## Important Notes
- **Community-Supported Only**: The content in this folder is **not officially supported** by Tenstorrent. While we encourage and appreciate community contributions, these distributions are maintained entirely by the community. Tenstorrent employees may still act as members of the community for the purposes of this folder.
- **Contributions Welcome**: We actively encourage contributions from the community to expand and improve the support for various Linux distributions. If you have a working setup for a distribution not yet represented here, consider sharing your work with the community by filing an issue and opening a pull request. 

## Disclaimer
The content in this folder is provided by the community and is not officially endorsed or supported by Tenstorrent. TO THE MAXIMUM EXTENT PERMITTED BY LAW, the scripts, installers, and guides provided here are delivered "as-is" without any warranty, expressed or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, or non-infringement. By using the materials in this folder, you acknowledge that you do so at your own risk and that Tenstorrent disclaims any liability for any damages or losses resulting from their use, including but not limited to direct, indirect, incidental, special, consequential, or punitive damages, or loss of data, business interruption, or any other commercial damages or losses. Tenstorrent reserves the right to remove a contribution at any time for any reason, though such removal does not revoke the license for copies already distributed under the Apache 2.0 license.

## Licensing
All contributions to this directory are subject to the licensing terms of the tt-metal repository. By submitting a pull request to this folder, contributors agree to license their code under the project's Apache 2.0 license as detailed in the repository's main LICENSE file, including all patent grant provisions contained therein. Contributors retain their copyright, but grant the project a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license to use, reproduce, modify, distribute, and sublicense the contribution. Any conflicting license statements in the contributed code will be superseded by this project's Apache 2.0 license.

### Contributor Certification
By submitting a contribution, you certify that:
1. You have the legal right to make the contribution and grant the licenses described above.
2. The contribution is your original work or you have sufficient rights from the copyright owner to submit it under the Apache 2.0 license.
3. The contribution does not, to your knowledge, violate any third party's copyrights, trademarks, patents, or other intellectual property rights.
4. You understand and agree that the contribution and record of it are public and maintained indefinitely.

### Indemnification
To the extent permitted by applicable law, contributors agree to indemnify, defend, and hold harmless Tenstorrent, its affiliates, officers, directors, employees, and agents from and against any and all claims, liabilities, damages, losses, costs, expenses, or fees (including reasonable attorneys' fees) arising from or related to: (a) your contribution; (b) your violation of these terms; or (c) your violation of any rights of another person or entity.
