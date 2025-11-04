TT-SMI
==================

Tenstorrent System Management Interface (TT-SMI) is a command line utility
to interact with all Tenstorrent devices on host.

Main objective of TT-SMI is to provide a simple and easy to use interface
to collect and display device, telemetry and firmware information.

In addition user can issue Grayskull board Tensix core reset.

Official Repository
--------

https://github.com/tenstorrent/tt-smi/

Getting Started
--------

Build and editing instruction are as follows -

Building from Git
--------

Install and source rust for the luwen library:

.. code-block::

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"

Optional
--------
Generate and source a python environment.  This is useful not only to isolate
your environment, but potentially easier to debug and use.  This environment
can be shared if you want to use a single environment for all your Tenstorrent
tools

.. code-block::

    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install --upgrade pip

Required
--------

Install TT-SMI.

.. code-block::

    pip3 install

Optional - For TT-SMI Developers
--------

Generate and source a python3 environment:

.. code-block::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install pre-commit

For users who would like to edit the code without re-building, install SMI in editable mode.

.. code-block::

    pip install --editable

Recommended: Install the pre-commit hooks so there is auto formatting for all files on committing.

.. code-block::

    pre-commit install

Usage
--------

Command line arguments:

.. code-block::

    tt-smi [-h] [-l] [-v] [-s] [-ls] [-f [snapshot filename]] [-g [GENERATE_RESET_JSON]] [-c]
                  [-r [0,1 ... or config.json ...]] [--snapshot_no_tty] [-glx_reset] [-glx_reset_auto]
                  [-glx_reset_tray {1,2,3,4}] [--no_reinit]

Getting Help
--------

Running TT-SMI with the ```-h, --help``` flag brings up the following output:

.. code-block::

    $ tt-smi --help

    usage: tt-smi [-h] [-l] [-v] [-s] [-ls] [-f [snapshot filename]] [-g [GENERATE_RESET_JSON]] [-c]
                  [-r [0,1 ... or config.json ...]] [--snapshot_no_tty] [-glx_reset] [-glx_reset_auto]
                  [-glx_reset_tray {1,2,3,4}] [--no_reinit]

    Tenstorrent System Management Interface (TT-SMI) is a command line utility to interact with all Tenstorrent devices
    on host. Main objective of TT-SMI is to provide a simple and easy to use interface to collect and display device,
    telemetry and firmware information. In addition user can issue Grayskull and Wormhole board level resets.

    options:
      -h, --help            show this help message and exit
      -l, --local           Run on local chips (Wormhole only)
      -v, --version         show program's version number and exit
      -s, --snapshot        Dump snapshot of current tt-smi information to STDOUT
      -ls, --list           List boards that are available on host and quits
      -f [snapshot filename], --filename [snapshot filename]
                            Write snapshot to a file. Default: ~/tt_smi/<timestamp>_snapshot.json
      -g [GENERATE_RESET_JSON], --generate_reset_json [GENERATE_RESET_JSON]
                            Generate default reset json file that reset consumes. Default stored at
                            ~/.config/tenstorrent/reset_config.json. Update the generated file and use it as an input
                            for the --reset option
      -c, --compact         Run in compact mode, hiding the sidebar and other static elements
      -r [0,1 ... or config.json ...], --reset [0,1 ... or config.json ...]
                            Provide list of PCI index or a json file with reset configs. Find PCI index of board using
                            the -ls option. Generate a default reset json file with the -g option.
      --snapshot_no_tty     Force no-tty behavior in the snapshot to stdout
      -glx_reset, --galaxy_6u_trays_reset
                            Reset all the asics on the galaxy host.
      -glx_reset_auto, --galaxy_6u_trays_reset_auto
                            Reset all the asics on the galaxy host, but do auto retries upto 3 times if reset fails.
      -glx_reset_tray {1,2,3,4}
                            Reset a specific tray on the galaxy.
      --no_reinit           Don't detect devices post reset.

Some flags are discussed in more detail in the following sections.

GUI
--------

To bring up the TT_SMI GUI:

.. code-block::

    $ tt-smi

The following display will appear:

.. image:: https://github.com/tenstorrent/tt-smi/blob/main/images/tt_smi.png

This is default mode displaying device information, telemetry, and firmware.

Latest SW Versions
--------

This section displays device software versions. If failures occur, error messages display the following:

.. image:: https://github.com/tenstorrent/tt-smi/blob/main/images/error.png

App Keyboard Shortcuts
--------

All app keyboard shortcuts can be found in the help menu. Bring up the help menu by pressing the  "h" or clicking the "help" button in the footer.

.. image:: https://github.com/tenstorrent/tt-smi/blob/main/images/help.png

Resets
--------

TT-SMI can perform resets on Blackhole, Wormhole and Grayskull PCIe cards:

.. code-block::

    -r/ --reset

PCIe configurations vary depending on the system. The current implementation of PCIe reset doesn't work as expected. We recommend a system reboot to reset a board in your system.

.. code-block::

    $ tt-smi -r 0,1 ... or config.json, --reset 0,1 ... or config.json

        Provide list of PCI index or a json file with reset configs. Find PCI index of board using the -ls option. Generate a default reset json file with the -g option.

To perform a reset, either provide a list of comma separated values of the PCI index of the cards on the host, or an input reset_config.json file that can be generated using:

.. code-block::

    -g/ --generate_reset_json

TT-SMI performs different reset types depending on the device:

- Grayskull: Resets each Tensix Core.
- Wormhole: Board level reset. Power is cut to the board and brought back up. Post reset the ethernet connections are re-trained.
- Blackhole: Board level reset. Power is cut to the board and brought back up.

By default, the reset command re-initializes boards after reset. To disable this, update the json config file.

A successful reset on a system with both Wormhole and Grayskull displays the following:

.. code-block::

    $ tt-smi -r 0,1

      Starting PCI link reset on WH devices at PCI indices: 1
      Finishing PCI link reset on WH devices at PCI indices: 1

      Starting Tensix reset on GS board at PCI index 0
      Lowering clks to safe value...
      Beginning reset sequence...
      Finishing reset sequence...
      Returning clks to original values...
      Finished Tensix reset on GS board at PCI index 0

      Re-initializing boards after reset....
     Done! Detected 3 boards on host.

OR

.. code-block::

    tt-smi -r reset_config.json

      Starting PCI link reset on WH devices at PCI indices: 1
      Finishing PCI link reset on WH devices at PCI indices: 1

      Starting Tensix reset on GS board at PCI index 0
      Lowering clks to safe value...
      Beginning reset sequence...
      Finishing reset sequence...
      Returning clks to original values...
      Finished Tensix reset on GS board at PCI index 0

      Re-initializing boards after reset....
      Done! Detected 3 boards on host.

To call the correct board dev ID for the reset, run the TT-SMI board list function: ``tt-smi -ls`` or ``tt-smi --list``. The dev ID listed is the same as found on: ``/dev/tenstorrent/<dev pci id>``.

The output includes a list of all boards on host and all boards that can be reset.

.. code-block::

    $ tt-smi -ls

    Gathering Information ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
                    All available boards on host:
    ┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ PCI Dev ID ┃ Board Type ┃ Device Series ┃ Board Number     ┃
    ┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ 0          │ Grayskull  │ e75           │ 0100007311523010 │
    │ 1          │ Wormhole   │ n300 L        │ 010001451170801d │
    │ N/A        │ Wormhole   │ n300 R        │ 010001451170801d │
    └────────────┴────────────┴───────────────┴──────────────────┘
                      Boards that can be reset:
    ┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ PCI Dev ID ┃ Board Type ┃ Device Series ┃ Board Number     ┃
    ┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ 0          │ Grayskull  │ e75           │ 0100007311523010 │
    │ 1          │ Wormhole   │ n300 L        │ 010001451170801d │
    └────────────┴────────────┴───────────────┴──────────────────┘

Disabling SW Version Reporting
--------

To disable software version and serial number reporting, update the following parameters in the reset config file. The reset file is generated:

.. code-block::

    ~/.config/tenstorrent/reset_config.json
    tt-smi -g

Run the following:

.. code-block::
  
     "disable_serial_report": false,     // make this true
     "disable_sw_version_report": false, // make this true

If the ``disable_sw_version_report`` is set to true, all the software versions in ``Latest SW Versions`` block are reported as "N/A".

Galaxy Resets
--------

The following methods reset Wormhole Galaxy 6U trays:
  - glx_reset: Resets the Galaxy, informs users of eth failures.
  - glx_reset_auto: Resets the Galaxy up to three times if eth failures are detected.
  - glx_reset_tray <tray_num>: Resets one galaxy tray. Tray numbers are 1-4.

Full Galaxy Reset:

.. code-block::

    tt-smi -glx_reset
     Resetting WH Galaxy trays with reset command...
    Executing command: sudo ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF
    Waiting for 30 seconds: 30
    Driver loaded
     Re-initializing boards after reset....
     Detected Chips: 32
     Re-initialized 32 boards after reset. Exiting...

Tray Reset:

.. code-block::

    tt-smi -glx_reset_tray 3 --no_reinit
     Resetting WH Galaxy trays with reset command...
    Executing command: sudo ipmitool raw 0x30 0x8B 0x4 0xFF 0x0 0xF
    Waiting for 30 seconds: 30
    Driver loaded
     Re-initializing boards after reset....
     Exiting after galaxy reset without re-initializing chips.

To identify the tray number to reset specific devices, run either ``tt-smi -glx_list_tray_to_device`` or ``tt-smi --galaxy_6u_list_tray_to_device``. These commands display a mapping table showing the relationship between tray numbers, tray bus IDs, and the corresponding PCI device IDs, making it easier to target a tray to reset. Do not run this command in a virtual machine (VM) environment, it requires direct hardware access to the Galaxy system.

.. code-block::

    $ tt-smi -glx_list_tray_to_device

        Gathering Information ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
              Mapping of trays to devices on the galaxy:
        ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Tray Number ┃ Tray Bus ID ┃ PCI Dev ID              ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ 1           │ 0xc0        │ 0,1,2,3,4,5,6,7         │
        │ 2           │ 0x80        │ 8,9,10,11,12,13,14,15   │
        │ 3           │ 0x00        │ 16,17,18,19,20,21,22,23 │
        │ 4           │ 0x40        │ 24,25,26,27,28,29,30,31 │
        └─────────────┴─────────────┴─────────────────────────┘

Snapshots
--------

TT-SMI provides an easy way to display information on the GUI in a json file, using the ``-s, --snapshot`` argument. By default the file is named and stored as `` ~/tt_smi/<timestamp>_snapshot.json``. Rename the file using the ``-f`` option. To not output a file, use ``tt-smi -f -``. It behaves like ``tt-smi -s``, printing snapshot info directly to STDOUT.

.. code-block::

    $ tt-smi -s -f tt_smi_example.json

        Gathering Information ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
          Saved tt-smi log to: tt_smi_example.json

.. code-block::

    $ tt-smi -f -

        Gathering Information ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
        {
            "time": "2025-02-04T13:04:50.313105",
            "host_info": {
                "OS": "Linux",
                "Distro": "Ubuntu 20.04.6 LTS",
                "Kernel": "5.15.0-130-generic",
            .........

License
--------

Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0.txt
