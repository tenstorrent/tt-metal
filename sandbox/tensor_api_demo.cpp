#include "ll_buda/host_api.hpp"
#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

Device *device = CreateDevice(DeviceType::GRAYSKULL, /*pci_express_slot=*/0);
Host *host = GetHost(); // TODO: Enable this

// Creates tensor on host with incremental values in row-major layout
// If this tensor is moved onto device, it get tilized and then stored in DRAM
// layout on host should be in tiles as well
Tensor a = Tensor(shape, Initialize::INCREMENT, Layout::TILES);


// Creates tensor on host with random data in row-major layout
// Allocates DRAM buffer on device
// Tilizes data and then moves data into DRAM buffer
Tensor b = Tensor(shape, Initialize::RANDOM, Layout::TILES, device);
// Tensor b is stored on device in tiled layout
/*
 - printing should be allowed in Row major and whatever current layout is
 - printing does NOT move tensor to host
1. New host only tensor object is created
2. device tensor is copied into the new host one
3. (potentially converts layout) and then prints it
4. New tensor object is destroyed
- device Tensor b is unchanged 
*/
b.print() // prints in layout


// Creates tensor on host with random data in row-major layout
// Allocates DRAM buffer on device
// Data remains in row major and then moves data into DRAM buffer
Tensor c = Tensor(shape, Initialize::RANDOM, Layout::ROW_MAJOR, device);


// DRAM loopback example (deallocation)
// This will deallocate the memory allocated from tensor d creation and d will be converted to host side tensor
Tensor d = Tensor(shape, Initialize::RANDOM, Layout::TILES, device);
d = d.to(host);

// DRAM loopback example (without deallocation)
// Tensor e does not get deallocated
// e and e_new are different unlinked tensors, changes in one will not be reflected in the other
Tensor e = Tensor(shape, Initialize::RANDOM, Layout::TILES, device);
Tensor e_new = e.to(host);


// Moves tensor f to host and then returns data as nested list 
Tensor f = Tensor(shape, Initialize::RANDOM, Layout::TILES, device);
std::vector<...> f_data = f.to_vec(); // get the data in a 1-D vector based on tensor layout 


// Moving host side tensor to device
Tensor g = Tensor(shape, Initialize::INCREMENT, Layout::TILES);
Tensor h = g.to(device) // Need to pass in the actual device 


// Data format conversion example - UNSUPPORTED ATM
/*
Tensor i = Tensor(shape, Initialize::RANDOM, Layout::TILES, DataFormat::BFloat16, device);
// are i and j the same tensor?
Tensor j = i.to(DataFormat::Float32)
*/

}  // namespace ll_buda

}  // namespace tt