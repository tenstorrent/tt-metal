Repro
=====

API index (autosummary table — row 6)
-------------------------------------

.. autosummary::

   ttnn_like.add
   ttnn_like.subtract
   ttnn_like.remainder

API page (field-list + dtypes table + example — rows 3/4/5)
-----------------------------------------------------------

.. autofunction:: ttnn_like.remainder

C++ page (breathe-shaped parameter table)
-----------------------------------------

.. cpp:function:: void tt_metal_launch(int device_id, int core_count)

   Launches a kernel on the given device.

   .. list-table::

      * - device_id
        - the device to launch on
      * - core_count
        - how many cores to use

Returns field in the form the ttnn pages emit
----------------------------------------------

.. py:function:: ttnn_like_op(a)

   Does a thing.

   :param a: the input tensor.
   :type a: ttnn.Tensor
   :returns: *ttnn.Tensor* -- the output tensor.
