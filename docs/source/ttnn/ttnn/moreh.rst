Moreh operators
###############

.. warning::

   ``ttnn.operations.moreh`` is deprecated.

   For new TT-NN code, prefer the regular ``ttnn`` API whenever an equivalent
   operator exists. Some ``moreh`` operators remain public because they cover
   backward or training-oriented paths, or because no clear regular ``ttnn``
   replacement exists today.

The ``moreh`` namespace remains available for compatibility and for a small
number of edge cases that still rely on it. The regular ``ttnn`` API is the
mainline surface and has the clearest public documentation coverage.

Use the table below as guidance when choosing between ``moreh`` and regular
``ttnn`` operators.

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - ``moreh`` operator family
     - Preferred direction
     - Notes
   * - ``moreh_sum``
     - ``ttnn.sum``
     - Prefer ``ttnn.sum`` in new reduction code.
   * - ``moreh_mean``
     - ``ttnn.mean``
     - Prefer ``ttnn.mean`` in new reduction code.
   * - ``moreh_layer_norm``
     - ``ttnn.layer_norm``
     - Prefer ``ttnn.layer_norm`` in new normalization code.
   * - ``moreh_group_norm``
     - ``ttnn.group_norm``
     - Prefer ``ttnn.group_norm`` in new normalization code.
   * - ``moreh_softmax``
     - ``ttnn.softmax``
     - Prefer ``ttnn.softmax`` in new softmax code.
   * - ``moreh_matmul``
     - ``ttnn.matmul``
     - Prefer ``ttnn.matmul`` in new matrix multiplication code.
   * - ``moreh_linear``
     - ``ttnn.linear``
     - Prefer ``ttnn.linear`` in new linear layers.
   * - ``moreh_cumsum``
     - ``ttnn.cumsum``
     - Prefer ``ttnn.cumsum`` where the regular API already covers the use
       case.
   * - ``moreh_norm``
     - No clear regular replacement today
     - Some tutorials and edge paths still rely on ``moreh_norm`` semantics.
       Use the regular ``ttnn`` API first, but do not assume there is a direct
       drop-in replacement here yet.
   * - ``*_backward`` families
     - Moreh-specific today
     - Many backward and training-oriented paths still live under
       ``ttnn.operations.moreh``.
   * - Optimizer-style families such as ``moreh_adam``, ``moreh_adamw``, and ``moreh_sgd``
     - Moreh-specific today
     - Treat these as specialized compatibility surfaces rather than the
       preferred public API.

Practical guidance
==================

1. Start with the regular ``ttnn`` API when an equivalent operator already
   exists in the main API reference.
2. Reach for ``ttnn.operations.moreh`` only when you need a moreh-specific
   backward path, a training-oriented compatibility surface, or an edge case
   that does not yet have a clear regular ``ttnn`` replacement.
3. If you are migrating older code, use the regular ``ttnn`` API wherever the
   mapping above is already clear and keep the remaining ``moreh`` calls narrow
   and intentional.
