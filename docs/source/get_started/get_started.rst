.. _Getting Started:

Getting Started
===============

The GitHub page for the project is located here:
https://github.com/tenstorrent-metal/tt-metal

To be added to the repo, please contact Jasmina or Davor at:
jvasiljevic@tenstorrent.com, dcapalija@tenstorrent.com.

Installation and environment setup instructions are in the GitHub repository
front-page README: https://github.com/tenstorrent-metal/tt-metal#installing

Running your first TT-Metal program
-----------------------------------

To run this program, ensure that you have cloned and
installed this project from source and activated and setup the necessary
environment. If you just came from the repo README after reading the source
installation and environment setup instructions, please continue.

Compile and execute an example application on Grayskull:

::

    make programming_examples/loopback
    ./build/programming_examples/loopback

Congratulations! You've run your first program on this stack. For more
explanation for what you just ran and a deeper dive into the API, please go to
:ref:`Dram Loopback<DRAM Loopback Example>`.

If you're an ML developer and looking for further docs for using the Python
library APIs to build models, please now go to :ref:`TT-Metal Models Get
Started`.

If you're an internal TT-Metal developer, please now read please review the
`contribution standards
<https://github.com/tenstorrent-metal/tt-metal/blob/main/CONTRIBUTING.md>`_.
