
## Why is this file in the experimental directory, if it doesn't use the experimental namespace?
mesh_socket.hpp is in the experimental directory because its APIs are expected to evolve significantly in the coming months, especiallyas we migrate from Rank based functionality to MeshId based functionality. But, since development of this header pre-dates the API stability policy, we've opted not to move it to the experimental namespace to minimize the disruption to existing user code.
