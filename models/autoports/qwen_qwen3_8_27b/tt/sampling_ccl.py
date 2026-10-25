# SPDX-License-Identifier: Apache-2.0
"""Persistent candidate gathers for the common Sampling1D callback contract."""
import ttnn


class SamplingCCL:
    def __init__(self,owner,*,persistent=True):
        self.owner=owner
        self.persistent=persistent
        self.buffers={}

    def __getattr__(self,name):
        return getattr(self.owner,name)

    def line_all_gather(self,tensor,*,dim,cluster_axis,memory_config,num_links,buffer_key):
        key=(buffer_key,tuple(tensor.shape),str(tensor.dtype))
        if self.persistent and key not in self.buffers:
            shape=list(tensor.shape);shape[dim]*=self.owner.mesh_device.get_num_devices()
            self.buffers[key]=ttnn.empty(shape,device=self.owner.mesh_device,dtype=tensor.dtype,
                layout=tensor.layout,memory_config=memory_config)
        gathered=ttnn.experimental.all_gather_async(tensor,dim=dim,persistent_output_buffer=self.buffers.get(key),
            multi_device_global_semaphore=self.owner.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.owner.get_and_cycle_barrier_semaphore_handle(),
            num_links=num_links,topology=ttnn.Topology.Ring,memory_config=memory_config,
            chunks_per_sync=10,num_workers_per_link=2,num_buffers_per_channel=2)
        # Sampling1D explicitly deallocates the gathered values. Return an owned
        # transient copy; the preallocated CCL backing buffer survives all traces.
        return ttnn.clone(gathered) if self.persistent else gathered
