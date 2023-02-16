

release_dst
===========

Releases the exclusive lock on the internal DST register for the current Tensix core. This lock had to be previously acquired with `acquire_dst`.
This call is blocking and is only available on the compute engine.

Return value: None

