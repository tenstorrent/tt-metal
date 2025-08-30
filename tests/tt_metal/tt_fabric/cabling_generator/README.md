
![](CablingFlow.png)
# Cabling Generator
A tool meant to be used in in scale-out of various Tenstorrent systems. Given 
deployment specifications (rack locations of systems i.e. from the rack 
elevation of a data center) and cabling specification (how to connect a set of
hosts in a certain topology) the cabling guide will generate a cutsheet listing
out each cabling link that a technician will need to attach.

## Expected Inputs
### Cabling Specification
This is where a topology expert will work. With no need to consider how the hosts are arranged physically, they can focus on the ideal way to connect a set of hosts together.

Functionally the Cabling Specification is a list of connections between 2 (host,tray,port) endpoints.

### Deployment Specification
This is where a person managing a specific data center deployment of a system cluster will work. After installing/setting up the hosts required for the cluster, the technician can fill out a deployment descriptor enumerating the physical location and hostnames of each host in the cluster they wish to connect.

### Putting Them Together
One thing to consider with how the Cabling Generator puts both the Specifications together is that the Cabling treats hosts indexes array, and the Deployment is basically an array of hosts. This is brought up to point out that order matters in the Deployment Specification; you will not get the same cabling guide if you mix up the order of hosts in a Deployment specification

<!-- ## Notes/Warnings -->
