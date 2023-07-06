## Installing

### From source on a Tenstorrent machine

Currently, the best way to use our software is through a
Tenstorrent-provisioned cloud machine and building from source.

Please use the communication information below if you'd like access.

0. If you're using a customer-facing cloud machine, SSH into the cloud machine:

```
ssh user@<external-ip> -p <ssh-port>
```

1. Create an SSH key for your machine.

```
ssh-keygen
```

2. Add the key to your Github profile. Please refer to [SSH keys on
   Github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

3. If you are an external customer, then you should already have a pre-cloned
version of the source code. Enter the repo:

```
cd tt-metal-user
```

If you do not have a pre-provisioned customer-facing machine, please continue
with this step to clone the repo.

Clone the repo.

```
git clone git@github.com:tenstorrent-metal/tt-metal.git --recurse-submodules
cd tt-metal
```

You should look ahead to [Getting started](#getting-started) to further build
and use this project.

### From a release wheel (BUDA-Eager only)

Coming soon!

## Getting started

After installing, please refer to our [Getting Started
page](https://tenstorrent-metal.github.io/tt-metal/latest/get_started/installation.html)
in our documentation.

## Documentation

Please refer to our
[documentation](https://tenstorrent-metal.github.io/tt-metal/latest/index.html).

## Contributing

We appreciate any contributions. Please review the [contributor's
guide](CONTRIBUTING.md) for more information.

## Communication

Announcements from the Tenstorrent team regarding this project will be in the
[discussions
page](https://github.com/orgs/tenstorrent-metal/discussions/categories/announcements).

If you have ideas you would like to bounce off others in the community before
formally proposing it, you can make a post in the [ideas discussions
page](https://github.com/orgs/tenstorrent-metal/discussions/categories/ideas).

If you would like to formally propose a new feature, report a bug, or have
issues with permissions, please through [GitHub
issues](https://github.com/tenstorrent-metal/tt-metal/issues/new/choose).
