# Getting up and running

0. If you're using a customer-facing cloud machine, SSH into the cloud machine:

```
ssh user@<external-ip> -p <ssh-port> -L 8888:localhost:8888
```

The ``-L`` option will be for docs later.

1. Create an SSH key for your machine.

```
ssh-keygen
```

2. Add the key to your Github profile. Please refer to [SSH keys on
   Github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

3. Clone the repo. If you received pre-provisioned customer-facing machine,
please skip this step.

```
git clone git@github.com:tenstorrent-metal/tt-metal.git --recurse-submodules
cd tt-metal
```

4. Build and activate the TT-Metal environment:
```
source ./build_tt_metal.sh
source build/python_env/bin/activate
```

5. Build the HTML Documentation page.

```
cd docs
make all
```

You can customize the port by using the `PORT=<port>` environment variable. If
you're using a customer-facing cloud machine, please disregard this point.

6. Navigate to the docs page.

Navigate your web browser to `http://<ip address>:<port>`, where `<ip address>`
is the IP address of the machine on which you launched the web server. For
example: `http://10.250.37.37:4242`, for port ``4242``.

If you're using a customer-facing cloud machine, navigate to
`http://localhost:8888`.

7. Follow the `Getting Started` instructions on the Documentation page.
