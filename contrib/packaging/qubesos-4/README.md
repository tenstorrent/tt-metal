# Installing the Tenstorrent stack on Qubes OS

This document describes how to install the Tenstorrent stack on Qubes OS.

The first few sections outline how to install Qubes OS from scratch, followed by an explanation for how to configure a Qubes VM for the Tenstorrent stack.

Users with an existing Qubes OS installation can skip ahead to [A Recap of Qubes VM configuration strategies](#a-recap-of-qubes-vm-configuration-strategies). This section discusses the different types of VMs available in Qubes OS and justifies the specific choices we make for how to install the Tenstorrent stack.

## Installing Qubes OS

For the initial install we follow the [Installation Guide](https://www.qubes-os.org/doc/installation-guide/) although we recommend customizing the installation options for a more streamlined experience:

### First boot (Installing base Qubes system)

- Disable storage encryption unless needed.

Note that depending on your hardware configuration, you may encounter a bug after the initial install when the system prepares to reboot. If you reach a black screen with the blinking cursor and the system does not automatically reboot, it is safe to manually reboot the system and continue to the next step.

### Second boot (Installing dom0 and templates)

- Disable `Debian` and `Whonix` TemplateVM installations. We will not use them.
- Disable `Create default application qubes` for (`personal`, `work`, `untrusted`, `vault`). We will not use them.
- Disable `Use a qube to hold all USB` unless required. _Enabling this is a common cause of installation failures_.

## Configuring Qubes OS

Once the Qubes installation is finished and we have logged in, we begin setting up the Tenstorrent stack.

### Update the Qube VMs

First, ensure the VMs are fully updated. For that, click the cube icon in the top right of the menu panel and select `Open Qube Manager`. You can also open this from the Qube menu at the top left of the panel.

In `Qube Manager`, select `dom0` and `fedora-<release>-xfce` and click the `Update` button in the top menu.

## A Recap of Qubes VM configuration strategies

Once updates have completed for `dom0` and `TemplateVM`, we begin setting up components of the Tenstorrent stack.

First, let us consider the available Qubes VM configuration strategies and how they influences our choices for setting up the Tenstorrent stack in a VM.

Qubes offers several distinct VM types: `TemplateVMs`, `AppVMs`, `StandaloneVMs` (and some other types which we can ignore for this guide).

These distinct VM types have different properties:

|         type | inherits root | persists root | persists home | network | passthrough | applications |
|--------------|---------------|---------------|---------------|---------|-------------|--------------|
|   TemplateVM |            no |           yes |           yes |      no |          no |  discouraged |
|        AppVM |           yes |            no |           yes |     yes |         yes |   encouraged |
| StandaloneVM |            no |           yes |           yes |     yes |         yes |   encouraged |

`TemplateVMs` persist the _entire filesystem_ but _are not configured_ for network access or for hardware passthrough and generally are not suitable for hosting applications.

`AppVMs` persist the _home directory_ exclusively but _are configured_ for network access and possible to configure for hardware passthrough and generally are suitable for hosting applications.

`StandaloneVMs` behave like a combination of `TemplateVMs` and `AppVMs`.

The usual recommendation from the Qubes project is to use `TemplateVMs` and `AppVMs` primarily, and this is how the initial installation is configured if you accept most of the default choices.

Unfortunately that approach is difficult to use for our Tenstorrent stack VM configuration due to the fact that we need 1) persistent root, and 2) network access at the same time in order to install some of the system components.

Normally `TemplateVMs` avoid this difficulty by installing everything through the system package manager, which is the one access point through which `TemplateVMs` do have network access.

But for the Tenstorrent stack, we need to install some system components which are not packaged by the system.

So the easiest approach is to use a `StandaloneVM`.

_The alternative, if you wish to explore it, is to either temporarily enable network access for the `TemplateVM` (which Qubes will warn you against), or use a separate `AppVM` to copy downloaded components to the `TemplateVM`, then install them through the `TemplateVM`, then shutdown the `TemplateVM`, then reboot the `AppVM` to re-inherit the changes, and finally continue with the `AppVM` as the primary VM._

## Configure a StandaloneVM

### Create the StandaloneVM

Next we prepare a `StandaloneVM` for installing the Tenstorrent stack.

Click the `New Qube` button in `Qube Manager`, then the `Basic` tab, and enter the following:

- `Name and label`: `tt` (color doesn't matter)
- `Type`: `StandaloneVM`
- `Template`: `fedora-<release>-xfce`
- `Networking`: `default (sys-firewall)`

Click `OK` to create the qube.

### Configure the StandaloneVM Storage

The default storage configuration for the provided VMs is on the small side so to ensure we do not run out of disk space during configuration we should increase the default settings.

Click the the `Basic` tab, then under `Disk storage`, modify the following:

- `Private storage max size` from `2.0 GiB` to `50.0 GiB`
- `System storage max size` from `20.0 GiB` to `250 GiB`

These are approximate values. Use what you have available, but ensure it will be enough to install and use the software.

Click `Apply` to save the new values.

### Configure the StandaloneVM Memory/CPU

Click the `Advanced` tab, then under `Memory/CPU`, modify the following:

- `Include in memory balancing`: uncheck since we are not running several VMs
- `Initial memory`: allocate most of your memory, leaving 4-8 GiB for the system.
- `Max memory`: ignore, since memory balancing is disabled
- `VCPUs`: allocate most of your cores, leaving 2-4 cores for the system.

Click `Apply` to save the new values.

### Configure the StandaloneVM Virtualization Mode

Click the `Advanced` tab, then under `Virtualization`, modify the following:

- `Mode`: `HVM`

The `HVM` mode is required in order to pass through devices like the `Tenstorrent Blackhole` to the VM.

### Configure the StandaloneVM Devices

Click the `Devices` tab, then under `Available Devices` on the left pane, select your Tenstorrent hardware, such as the `Blackhole`. Next click the `>` button to enable hardware passthrough and make the device available to the `StandaloneVM`.

On many systems this should be enough for hardware passthrough to work. Additional settings may be required if you encounter errors. Consult the Qubes OS documentation for more hardware passthrough settings if necessary.

### Adjust the StandaloneVM `qrexec_timeout`

Before starting the `tt` VM we should also change the `qrexec_timeout` setting in order to avoid another common point of failure when setting up Qubes.

The `qrexec_timeout` value influences how long the system waits when starting up a VM and if the value is too low, the system will report a VM launch failure and prevent the VM from launching.

Because we resized the storage for the VM, the system will perform the resize operation when the VM is next launched, and this may cause a timeout with the default `qrexec_timeout` setting.

In order to change `qrexec_timeout`, open a terminal in `dom0`.

This can be done by clicking on the menu icon in the top left of the menu panel, then clicking the `settings` gear icon on the left of the menu, then clicking `other`, and then clicking `Xfce Terminal` from the application icons on the right of the menu. Alternatively, you can enter type `dom0 term` into the search field and select the corresponding `Xfce Terminal` from the filtered entries.

From the `dom0` terminal prompt run the following command:

```bash
qvm-prefs tt qrexec_timeout 3600
```

This increases the timeout to 1 hour, but in general the resize operation should complete in at most a couple of minutes.

### Launch the StandaloneVM terminal and run the Tenstorrent installer

From the `Qube Manager` window (re-open from the blue cube icon in the top right of the menu panel if needed), right click on the `fedora-<release>-xfce` TemplateVM and click `Start/Resume`.

After a moment, you should get a notification that the TemplateVM has started and see the green dot under `State`.

Next, open the `Xfce Terminal` for `fedora-<release>-xfce`, similar to how you opened it for `dom0`: click on the menu icon in the top level of the menubar panel, then click the `Templates` tab up top, then select `fedora-<release>-xfce` from the list on the left, then `Xfce Terminal` on the right. Alternatively you can type `fedora term` into the search field and select the corresponding `Xfce Terminal` from the filtered entries.

### Install the Tenstorrent system components

From the `fedora-<release>-xfce` terminal, prepare to execute a series of commands:

Fetch the Tenstorrent installer from GitHub:

```bash
mkdir -p ~/Downloads/tt
cd ~/Downloads/tt
curl -fsSLO https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh
chmod +x ./install.sh
```

Execute the installer with the following options. When prompted, select the default `new-venv` for the Python choice.

```bash
./install.sh \
  --install-kmd \
  --install-hugepages \
  --install-sfpi \
  --no-install-podman \
  --no-update-firmware \
  --python-choice=new-venv
```

The Tenstorrent installer configures the `tenstorrent` kernel module for the latest Fedora kernel, which is later than the most recently supported kernel for the Fedora Qubes VM, so we need to tell DKMS to recompile the module for the actual kernel we are using:

```bash
sudo dkms install \
  -m tenstorrent \
  -v $(sudo dkms status | awk -F'[/,]' '$1 == "tenstorrent" {print $2}' | sort -r | head -1) \
  -k $(uname -r)
```

Finally, reboot the system before continuing. For this, right click the `tt` VM in the `Qube Manager` and select `Restart`. This may again take a minute or two before the VM becomes responsive (even after the green dot appears under status). Similar to before, launch the `Xfce Terminal` and once it opens we will continue.

## Check that your Tenstorrent hardware is visible in the HVM

From the `Xfce Terminal` check that your Tenstorrent hardware is visible and working.

First, check `lspci`:

```bash
lspci -vvv
```

You should see some output like this if everything is working correctly:

```text
00:07.0 Processing accelerators: Tenstorrent Inc Blackhole
        Subsystem: Tenstorrent Inc Device 0040
        Kernel driver in use: tenstorrent
        Kernel module: tenstorrent
```

Next check that the device shows up under `tt-smi`:

```bash
source ~/.tenstorrent-venv/bin/activate
tt-smi
```

You should see the TT-SMI terminal UI load. Press (2) to check the Telemetry tab and ensure the readouts look normal.

Then, exit the `.tenstorrent-venv` virtualenv. We will create a separate one for the `ttnn` installation next:

```bash
exit
```

## Configure Python and install `ttnn` and `torch`

From the `Xfce Terminal`, create a directory for the Python virtual environment we will be using:

```bash
mkdir -p ~/Development/tt
cd ~/Development/tt
```

Next, install Python 3.10:

```bash
sudo dnf install python3.10
```

Create a Python 3.10 virtualenv and install `ttnn`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install ttnn
```

Next, while still in the virtualenv, install `torch`. Note you will probably have to override `TMPDIR` otherwise you get an error about lack of space:

```bash
TMPDIR=/var/tmp pip install torch
```

## Run a `ttnn` example

At this point everything should be set up and we can run a basic `ttnn` example to check that the stack is working.

Go to [basic examples](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) page for reference.

Open a python REPL and type or paste the code below. You should see output from the system indicating activity from the Tenstorrent hardware.

```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_input_tensor_a = torch.rand(4, 7, dtype=torch.float32)
input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output_tensor = ttnn.exp(input_tensor_a)
torch_output_tensor = ttnn.to_torch(output_tensor)

torch_input_tensor_b = torch.rand(7, 1, dtype=torch.float32)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
matmul_output_tensor = input_tensor_a @ input_tensor_b
torch_matmul_output_tensor = ttnn.to_torch(matmul_output_tensor)

print(torch_matmul_output_tensor)

ttnn.close_device(device)
```

With that, you now have a VM configured for hardware passthrough for your Tenstorrent device, capable of running the Tenstorrent software stack.

Additional software can be installed from Fedora packages in the usual way. If extra storage is needed, you can shutdown the VM and increase the disk size, as before, then restart.
