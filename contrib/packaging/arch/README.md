# Installing the Tenstorrent stack on Arch Linux

This document describes how to install the Tenstorrent stack on Arch Linux.

This article covers installation of the Tenstorrent software stack on Arch Linux, including kernel modules, utilities, and the TTNN neural network library.

> [!NOTE]
> Though as of writing this document, the maintainer of the AUR packages is a Tenstorrent employee, these packages should be considered community-supported. Tenstorrent does not officially support Arch Linux. And the packages are maintained in an best-effort basis.

## Prerequisites

An [AUR helper](/index.php/AUR_helpers) is required to install packages from the [AUR](/index.php/Arch_User_Repository). This guide uses [yay](/index.php/Yay).

### Installing yay

**Note:** If you already have an AUR helper installed, skip this section.

Install the required dependencies:

```bash
sudo pacman -S git base-devel
```

Clone and build `yay`:

```bash
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
```

## Installation

### Base packages

The core Tenstorrent stack packages available in the AUR:

- `tt-kmd-dkms` - Kernel module for device communication
- `tt-smi` - System management interface
- `tt-flash` - Firmware flashing utility
- `tt-topology` - Device topology detection
- `tt-burnin` - Hardware burn-in testing (optional)

Install the packages:

```bash
yay -S tt-kmd-dkms tt-smi tt-flash tt-topology tt-burnin
```

### Development packages

For the latest development versions, `-git` variants are available:

```bash
yay -S tt-kmd-git-dkms tt-smi-git tt-flash-git
```

> [!WARNING]
> The `tt-kmd-git-dkms` package does not automatically rebuild when the kernel is updated. After kernel updates, manually trigger a rebuild with `dkms` or reinstall the package. This is intended to avoid upgrading causing breakages and leaves the system in a usable state.

## Configuration

### Loading the kernel module

Since the kernel module is installed via [DKMS](/index.php/Dynamic_Kernel_Module_Support), reboot to load it:

```bash
sudo reboot
```

### Verification

Check that Tenstorrent devices are detected:

```bash
ls /dev/tenstorrent/*
```

This lists devices directly connected via PCIe.

## TTNN

[TTNN](https://github.com/Tenstorrent/tt-metal) is Tenstorrent's neural network operator and tensor library.

### Installation

To install the system-wide Python package:

```bash
yay -S python-ttnn
```

> [!NOTE]
> This is a large package with extended build times.

> [!TIP]
> Developers modifying TTNN should compile from source instead. See the [upstream installation guide](/INSTALLING.md).

### Testing

Verify the installation with a basic TTNN example:

```bash
python
```

```python
import ttnn

device = ttnn.open_device(device_id=0)
a = ttnn.rand((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(ttnn.sigmoid(a))
```

A tensor should be printed to the console, confirming the stack is working correctly.

## See also

* [Tenstorrent documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples)
* [TTNN repository](https://github.com/Tenstorrent/tt-metal)
