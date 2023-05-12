TT-Dispatch
***********

Overview
========

TT-Dispatch is Python based framework providing users with simple interface to
run models on TT-Metal Platform via first-party developed ttTensor and ttDNN libraries.

TT-Dispatch will provide implementations of various models used in production
systems and ongoing efforts of partly-supported models.

TT Device
=========

TT Device is a library used to interact with Tenstorrent accelerator device.
The host machine executing Python code is described by class Host, while Tenstorrent accelerator device is described by class Device.

The following example shows how to create, initialize, and close TT accelerator device.

.. code-block:: python

    from libs import tt_lib as tt_lib

    # Create an instance of TT accelerator device.
    # This is device of type GRAYSKULL and is attached to PCIexpress slot number 0 on host.
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)

    # Initialize the instance of TT accelerator device before it can be used.
    tt_lib.device.InitializeDevice(device)

    # Run models on TT accelerator device
    ...

    # Close TT accelerator device after finished using it.
    tt_lib.device.CloseDevice(device)

Known Issues
============
Python garbage collection sometimes garbage collects a created device before tensors that
have buffers on device. This results in a segmentation fault at the end of the scope where
the device / tensors were created.

Current Workarounds:

- Create device in an outer scope compared to tensors on device, ex. global scope

.. code-block:: python

    from libs import tt_lib as tt_lib

    # Create an instance of TT accelerator device.
    # This is device of type GRAYSKULL and is attached to PCIexpress slot number 0 on host.
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)

    def compute(device):
        # Initialize the instance of TT accelerator device before it can be used.
        tt_lib.device.InitializeDevice(device)

        # Run models on TT accelerator device
        ...

        # Close TT accelerator device after finished using it.
        tt_lib.device.CloseDevice(device)

    compute(device)

- Delete/Dereference any on device tensors returned from ops in the same scope as device

.. code-block:: python

    from libs import tt_lib as tt_lib

    # Create an instance of TT accelerator device.
    # This is device of type GRAYSKULL and is attached to PCIexpress slot number 0 on host.
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)

    # Initialize the instance of TT accelerator device before it can be used.
    tt_lib.device.InitializeDevice(device)

    # Run models on TT accelerator device
    ...
    # Option 1: Delete reference outright
    out = tt_lib.tensor.op(args)
    out_host = out.to(host)
    del out

    # Option 2: Lose reference
    out = tt_lib.tensor.op(args).to(host)

    # Close TT accelerator device after finished using it.
    tt_lib.device.CloseDevice(device)

----

.. autoclass:: tt_lib.device.Host
    :members:
    :special-members: __init__

.. autoclass:: tt_lib.device.Device
    :members:
    :special-members: __init__

.. autoclass:: tt_lib.device.Arch
    :members: GRAYSKULL

.. autofunction:: tt_lib.device.CreateDevice

.. autofunction:: tt_lib.device.InitializeDevice

.. autofunction:: tt_lib.device.GetHost

.. autofunction:: tt_lib.device.CloseDevice

.. autofunction:: tt_lib.device.SetDefaultDevice

.. autofunction:: tt_lib.device.GetDefaultDevice



Models statuses
===============

To come.
