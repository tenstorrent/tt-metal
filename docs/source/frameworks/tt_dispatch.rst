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
The host machine executing Python code is described by class Host, while Tenstorrent accellerator device is described by class Device.

The following example shows how to create, initialize, and close TT accelerator device.

.. code-block:: python

    from pymetal import ttlib as ttlib

    # Create an instance of TT accelerator device.
    # This is device of type GRAYSKULL and is attached to PCIexpress slot number 0 on host.
    device = ttlib.device.CreateDevice(ttlib.device.Arch.GRAYSKULL, 0)
    
    # Initialize the instance of TT accelerator device before it can be used.
    ttlib.device.InitializeDevice(device)
    
    # Run models on TT accelerator device
    ...

    # Close TT accelerator device after finished using it.
    ttlib.device.CloseDevice(device)



.. autoclass:: ttlib.device.Host
    :members:
    :special-members: __init__

.. autoclass:: ttlib.device.Device
    :members:
    :special-members: __init__

.. autoclass:: ttlib.device.Arch
    :members: GRAYSKULL

.. autofunction:: ttlib.device.CreateDevice

.. autofunction:: ttlib.device.InitializeDevice

.. autofunction:: ttlib.device.GetHost

.. autofunction:: ttlib.device.CloseDevice



Models statuses
===============

To come.
