# Community-Driven Linux Distribution Support

Welcome to the community-driven Linux distribution support folder for Tenstorrent's tt-metal repository! This folder is dedicated to providing scripts, installers, packaging, and guides for getting Tenstorrent software and hardware working on various Linux distributions beyond what Tenstorrent officially supports. While Tenstorrent does not officially support these distributions, we believe in fostering collaboration and resource-sharing among our users. Thank you to all contributors who help make this possible!

## Folder Structure
The folder structure is organized as follows:
tt-metal/contrib/packaging/[DISTRO]/...

Where `[DISTRO]` is replaced with the name and version of the Linux distribution (e.g., `Debian12`, `Fedora42`, `ArchLinux`, etc.). Each distribution-specific folder will contain relevant scripts, installers, and/or guides contributed by the community.

## Contributing
If you would like to contribute to this effort, please review the following guidelines:
0. Check our issue tracker for existing work related to your distribution.
1. Create a new folder under `tt-metal.git/contrib/packaging/` for your Linux distribution if one does not already exist.
2. Include detailed documentation (e.g., a `README.md` file) within your distribution folder that explains how to use the provided scripts or installers.
3. Ensure that any scripts or installers you provide are well-commented and follow best practices for security and maintainability.
4. Submit your contribution via a pull request for review by the community.

## Important Notes
- **Community-Supported Only**: The content in this folder is **not officially supported** by Tenstorrent. While we encourage and appreciate community contributions, these distributions are maintained entirely by the community. Tenstorrent employees may still act as members of the community for the purposes of this folder.
- **Contributions Welcome**: We actively encourage contributions from the community to expand and improve the support for various Linux distributions. If you have a working setup for a distribution not yet represented here, consider sharing your work with the community by filing an issue and opening a pull request.
- **Use at Your Own Risk**: The scripts, installers, and guides provided here are delivered "as-is" without any warranty, expressed or implied. If something breaks, it is the responsibility of the community to address and resolve the issue.

## Disclaimer
The content in this folder is provided by the community and is not officially endorsed or supported by Tenstorrent. By using the materials in this folder, you acknowledge that you do so at your own risk and that Tenstorrent disclaims any liability for damages or losses resulting from their use.

## Licensing
All contributions to this folder are subject to the licensing terms of the tt-metal repository. For more details, please refer to the repository's main [LICENSE](https://github.com/tenstorrent/tt-metal/blob/main/LICENSE) file.
