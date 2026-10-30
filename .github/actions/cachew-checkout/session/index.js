// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Session lifecycle for cachew checkout.
//
// Files in the action-owned session directory under RUNNER_TEMP:
//   global.config    The job's GIT_CONFIG_GLOBAL after activation. Starts as a
//                    copy of the caller's global settings (inherited cachew
//                    rewrites removed). User `git config --global` writes land
//                    here and are never discarded; cachew only adds and removes
//                    its own keys (state.managed) in place, through Git's lock.
//   checkout.config  GIT_CONFIG_GLOBAL while actions/checkout runs. Includes
//                    global.config, so checkout sees the caller's settings, while
//                    checkout's temporary global writes (submodule auth, its
//                    transient safe.directory) stay in this throwaway file.
//   state.json       Session state, including cache-scoped auth headers.
//
// Upstream (github.com) authentication is never added to the global file:
// actions/checkout already persists repository-scoped credentials, and a second
// job-wide Authorization header makes GitHub reject requests as duplicates.

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const cp = require('node:child_process');

const TRUSTED_CACHES = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'trusted_caches.json'), 'utf8')).origins;
const PERFORMANCE = [
  ['checkout.workers', '0'],
  ['checkout.thresholdForParallelism', '50'],
  ['submodule.fetchJobs', '0'],
  ['gc.auto', '0'],
];
// GIT_CONFIG_GLOBAL needs Git 2.32. Older Git would silently ignore the isolation.
const MINIMUM_GIT = [2, 32];

function fileCommand(file, key, value) {
  if (!file) return;
  const delimiter = `cachew_${crypto.randomUUID()}`;
  fs.appendFileSync(file, `${key}<<${delimiter}\n${value}\n${delimiter}\n`);
}

function output(key, value) { fileCommand(process.env.GITHUB_OUTPUT, key, value); }
function environment(key, value) { fileCommand(process.env.GITHUB_ENV, key, value); }
function save(key, value) { fileCommand(process.env.GITHUB_STATE, key, value); }

function mask(value) {
  if (value) console.log(`::add-mask::${value.replaceAll('%', '%25').replaceAll('\r', '%0D').replaceAll('\n', '%0A')}`);
}

function executable(name) {
  for (const directory of (process.env.PATH || '').split(path.delimiter)) {
    if (!directory) continue;
    // Never resolve Git to an earlier session's wrapper.
    if (name === 'git' && path.basename(directory) === 'bin' && path.basename(path.dirname(directory)).startsWith('cachew-')) continue;
    const candidate = path.join(directory, name);
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      return fs.realpathSync(candidate);
    } catch {}
  }
  throw new Error(`Required executable unavailable: ${name}`);
}

function quote(value) {
  return `"${String(value).replaceAll('\\', '\\\\').replaceAll('"', '\\"').replaceAll('\n', '\\n').replaceAll('\r', '\\n').replaceAll('\t', '\\t')}"`;
}

function entry(key, value) {
  const first = key.indexOf('.');
  const last = key.lastIndexOf('.');
  if (first < 1) throw new Error('Invalid Git configuration key');
  const section = key.slice(0, first);
  const subsection = first === last ? '' : ` ${quote(key.slice(first + 1, last))}`;
  return `[${section}${subsection}]\n\t${key.slice(last + 1)} = ${quote(value)}\n`;
}

function writePrivate(file, content) {
  const temporary = `${file}.${crypto.randomUUID()}`;
  fs.writeFileSync(temporary, content, {mode: 0o600});
  fs.renameSync(temporary, file);
}

function writeState(directory, state) {
  writePrivate(path.join(directory, 'state.json'), JSON.stringify(state));
}

function readState(directory) {
  const root = fs.realpathSync(process.env.RUNNER_TEMP);
  const resolved = fs.realpathSync(directory);
  if (path.dirname(resolved) !== root || !path.basename(resolved).startsWith('cachew-')) {
    throw new Error('Invalid cachew session directory');
  }
  return JSON.parse(fs.readFileSync(path.join(resolved, 'state.json'), 'utf8'));
}

// Plain HTTP is only for the loopback fixtures in tests/. It is enabled by a
// test-only variable that action.yml never forwards (it clears it), so no
// workflow input can select plain HTTP, not even to 127.0.0.1.
function loopbackHttpAllowed() { return process.env.CACHEW_TEST_ALLOW_LOOPBACK_HTTP === '1'; }

function origin(value) {
  const url = new URL(value);
  const loopback = loopbackHttpAllowed() && url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname);
  if ((!loopback && url.protocol !== 'https:') || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
    throw new Error('Unsupported Git origin');
  }
  return url.origin;
}

function settings() {
  let upstream = 'https://github.com';
  let caches = [];
  let upstreamValid = true;
  let valid = true;
  try { upstream = origin(process.env.CACHEW_GITHUB_SERVER_URL || process.env.GITHUB_SERVER_URL); }
  catch { upstreamValid = false; valid = false; }
  const requested = (process.env.CACHEW_CACHEW_URL || 'auto').trim();
  if (requested.toLowerCase() === 'auto') caches = [...TRUSTED_CACHES];
  else {
    try {
      const cache = origin(requested);
      if (TRUSTED_CACHES.includes(cache) || (loopbackHttpAllowed() && upstream.startsWith('http:') && cache.startsWith('http:'))) caches = [cache];
    } catch {}
  }
  if (!caches.length) valid = false;
  const repository = process.env.CACHEW_REPOSITORY || process.env.GITHUB_REPOSITORY || '';
  const repositoryValid = /^[\w.-]+\/[\w.-]+$/.test(repository) && !repository.split('/').some(part => ['.', '..'].includes(part));
  if (!repositoryValid) valid = false;
  let timeout = Number(process.env.CACHEW_CACHEW_FETCH_TIMEOUT);
  let healthTimeout = Number(process.env.CACHEW_CACHEW_HEALTH_TIMEOUT);
  if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 600 || !Number.isFinite(healthTimeout) || healthTimeout <= 0 || healthTimeout > 30) {
    valid = false;
    timeout = 120;
    healthTimeout = 2;
  }
  return {upstream, caches, repository, timeout, healthTimeout, valid, upstreamValid, repositoryValid};
}

function cleanEnvironment() {
  const env = {...process.env};
  for (const key of Object.keys(env)) if (key.startsWith('GIT_CONFIG_') || key.startsWith('GIT_TRACE')) delete env[key];
  return env;
}

function run(realGit, args, allowed = [0]) {
  const result = cp.spawnSync(realGit, args, {env: cleanEnvironment(), encoding: 'utf8'});
  if (!allowed.includes(result.status)) throw new Error('Git configuration command failed');
  return result;
}

function git(state, args) { return run(state.realGit, args).stdout; }

function gitVersionSupported(realGit) {
  const match = /(\d+)\.(\d+)/.exec(run(realGit, ['version']).stdout);
  if (!match) return false;
  const [major, minor] = [Number(match[1]), Number(match[2])];
  return major > MINIMUM_GIT[0] || (major === MINIMUM_GIT[0] && minor >= MINIMUM_GIT[1]);
}

function isCacheHost(value, selected) {
  try {
    const url = new URL(value);
    return selected.includes(url.origin) || url.hostname === 'cachew' || url.hostname.startsWith('cachew.');
  } catch { return false; }
}

function cacheTarget(value, selected) {
  try { return new URL(value).pathname.startsWith('/git/') && isCacheHost(value, selected); } catch { return false; }
}

// Copy the caller's global configuration, following includes, without any
// inherited cachew rewrite or cache credential.
function copyGlobals(directory, state, caches) {
  if (state.originalGlobal === '') return '';
  const copies = new Map();
  let counter = 0;
  const originalHome = process.env.HOME || os.homedir();
  function copy(source) {
    source = path.resolve(source);
    if (copies.has(source)) return copies.get(source);
    if (!fs.existsSync(source)) return null;
    if (copies.size > 30) throw new Error('Too many included Git configuration files');
    const destination = path.join(directory, `base-${counter++}.config`);
    copies.set(source, destination);
    const contents = git(state, ['config', '--file', source, '--no-includes', '--null', '--list']);
    let result = '';
    for (const item of contents.split('\0')) {
      if (!item) continue;
      const separator = item.indexOf('\n');
      let key = separator < 0 ? item : item.slice(0, separator);
      let value = separator < 0 ? 'true' : item.slice(separator + 1);
      const lower = key.toLowerCase();
      if (lower.startsWith('url.') && lower.endsWith('.insteadof') && cacheTarget(key.slice(4, -10), caches)) continue;
      if (lower.startsWith('http.') && lower.endsWith('.extraheader') && isCacheHost(key.slice(5, -12), caches)) continue;
      if (lower === 'include.path' || /^includeif\..*\.path$/.test(lower)) {
        const included = value.startsWith('~/') ? path.join(originalHome, value.slice(2)) : path.resolve(path.dirname(source), value);
        const copied = copy(included);
        if (!copied) continue;
        value = copied;
        key = key.replace(/^(includeif\.gitdir(?:\/i)?:)(\.\/|~\/)/i, (_, prefix, type) => `${prefix}${type === './' ? path.dirname(source) : originalHome}/`);
      }
      result += entry(key, value);
    }
    fs.writeFileSync(destination, result, {mode: 0o600});
    return destination;
  }
  const originals = state.originalGlobal !== null ? [state.originalGlobal] : state.originalPaths;
  return originals.map(copy).filter(Boolean).map(file => entry('include.path', file)).join('');
}

function removeManaged(state, file) {
  const keys = [...new Set((state.managed || []).map(([key]) => key))];
  for (const key of keys) run(state.realGit, ['config', '--file', file, '--unset-all', key], [0, 5]);
  state.managed = [];
}

function addManaged(state, file, entries) {
  // Record first, so a partial failure is still removed by the next cleanup.
  state.managed = entries;
  for (const [key, value] of entries) run(state.realGit, ['config', '--file', file, '--add', key, value]);
}

function getAll(state, file, key) {
  if (!fs.existsSync(file)) return [];
  const result = run(state.realGit, ['config', '--file', file, '--no-includes', '--null', '--get-all', key], [0, 1]);
  return result.status === 0 ? result.stdout.split('\0').slice(0, -1) : [];
}

function writeCheckoutConfig(directory) {
  writePrivate(path.join(directory, 'checkout.config'), entry('include.path', path.join(directory, 'global.config')));
}

function disableInPlace(directory, state) {
  fs.writeFileSync(path.join(directory, 'cache-disabled'), '', {mode: 0o600});
  removeManaged(state, path.join(directory, 'global.config'));
  state.active = false;
}

function prepare() {
  const config = settings();
  const existing = process.env.CACHEW_SESSION_DIR;
  if (existing && fs.existsSync(existing)) {
    // A later checkout in the same job: remove the cache rewrite in place,
    // keeping user writes, so the official checkout talks to upstream.
    const state = readState(existing);
    try {
      removeManaged(state, path.join(existing, 'global.config'));
      state.active = false;
      writeCheckoutConfig(existing);
      writeState(existing, state);
    } catch (error) {
      try { disableInPlace(existing, state); writeState(existing, state); } catch {}
      throw error;
    }
    return {directory: existing, state, created: false};
  }
  // An unusable or untrusted cache/upstream origin means no session at all:
  // plain checkout, no token anywhere, no wrapper and no configuration change.
  if (!config.valid) throw new Error('Unsupported cachew configuration');
  const realGit = executable('git');
  if (!gitVersionSupported(realGit)) throw new Error('Git 2.32 or newer is required');
  const directory = fs.mkdtempSync(path.join(fs.realpathSync(process.env.RUNNER_TEMP), 'cachew-'));
  try {
    fs.chmodSync(directory, 0o700);
    const originalHome = process.env.HOME || os.homedir();
    const state = {
      version: 2,
      active: false,
      realGit,
      originalGlobal: process.env.GIT_CONFIG_GLOBAL ?? null,
      originalPaths: [path.join(process.env.XDG_CONFIG_HOME || path.join(originalHome, '.config'), 'git', 'config'), path.join(originalHome, '.gitconfig')],
      tokens: {},
      defaults: {},
      managed: [],
      cache: null,
    };
    const base = copyGlobals(directory, state, config.caches);
    writePrivate(path.join(directory, 'global.config'), base + PERFORMANCE.map(([key, value]) => entry(key, value)).join(''));
    writeCheckoutConfig(directory);
    writeState(directory, state);
    return {directory, state, created: true};
  } catch (error) {
    fs.rmSync(directory, {recursive: true, force: true});
    throw error;
  }
}

function prepareMain() {
  // Marks the registered post invocation, which must never prepare again.
  save('cachew-post', 'true');
  let prepared;
  try {
    prepared = prepare();
  } catch {
    // Never fail a checkout that plain actions/checkout could perform. When a
    // new session cannot be created the job's Git configuration is untouched.
    output('ready', 'false');
    console.log('Cachew unavailable on this runner; using standard checkout only.');
    return;
  }
  const {directory, state, created} = prepared;
  // Only the creating invocation's post removes the session. Posts run in
  // reverse order, so it is the last cachew post of the job.
  if (created) save('cachew-session', directory);
  output('session-dir', directory);
  output('ready', 'true');
  environment('CACHEW_SESSION_DIR', directory);
  environment('CACHEW_REAL_GIT', state.realGit);
  environment('CACHEW_NODE', process.execPath);
  environment('CACHEW_CONFIG_HELPER', __filename);
  environment('GIT_CONFIG_GLOBAL', path.join(directory, 'checkout.config'));
}

function authEntries(url, header) {
  // The empty value resets less specific headers, so exactly one is sent.
  return [[`http.${url}/.extraHeader`, ''], [`http.${url}/.extraHeader`, header]];
}

function healthy(cache, timeout) {
  const result = cp.spawnSync('curl', ['--disable', '--silent', '--fail', '--retry', '0', '--max-time', String(timeout), '--output', os.devnull, '--write-out', '%{http_code}', cache + '/_liveness'], {timeout: (timeout + 0.5) * 1000});
  return result.status === 0 && result.stdout?.toString() === '200';
}

function shellQuote(value) { return `'${value.replaceAll("'", "'\\''")}'`; }

// Only commands that can reach a remote pay for the Python wrapper.
function shim(state, directory, python, wrapper) {
  return `#!/bin/sh
export CACHEW_REAL_GIT=${shellQuote(state.realGit)}
export CACHEW_SESSION_DIR=${shellQuote(directory)}
skip=0
for argument in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  case "$argument" in
    -C|-c|--git-dir|--work-tree|--namespace|--config-env|--super-prefix) skip=1 ;;
    -*) ;;
    clone|fetch|ls-remote|push|pull|submodule|remote) exec ${shellQuote(python)} ${shellQuote(wrapper)} "$@" ;;
    *) break ;;
  esac
done
exec "$CACHEW_REAL_GIT" "$@"
`;
}

function activate() {
  const directory = process.env.CACHEW_SESSION_DIR;
  const state = readState(directory);
  const global = path.join(directory, 'global.config');
  const config = settings();
  const token = process.env.CACHEW_TOKEN || '';
  const header = `AUTHORIZATION: basic ${Buffer.from(`x-access-token:${token}`).toString('base64')}`;
  mask(token);
  mask(header.slice('AUTHORIZATION: basic '.length));
  if (token && config.upstreamValid && config.repositoryValid && (process.env.CACHEW_PERSIST_CREDENTIALS || 'true').toLowerCase() !== 'false') {
    // Cache-scoped only. Upstream credentials stay where actions/checkout put them.
    state.tokens[`${config.upstream}/${config.repository}`] = header;
    if (config.repository.toLowerCase() === (process.env.GITHUB_REPOSITORY || '').toLowerCase()) state.defaults[config.upstream] = header;
  }
  // Keep checkout's safe.directory entries (written to checkout.config) for later steps.
  // An empty value resets the list in Git, so only entries after the last reset count.
  const trusted = [];
  for (const value of getAll(state, path.join(directory, 'checkout.config'), 'safe.directory')) {
    if (value === '') trusted.length = 0;
    if (!trusted.includes(value)) trusted.push(value);
  }
  const present = getAll(state, global, 'safe.directory');
  for (const value of trusted) {
    if (value !== '' && present.includes(value)) continue;
    run(state.realGit, ['config', '--file', global, '--add', 'safe.directory', value]);
    present.push(value);
  }
  const disabled = path.join(directory, 'cache-disabled');
  let python;
  try {
    python = executable('python3');
    if (cp.spawnSync(python, ['-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)']).status !== 0) python = null;
  } catch {}
  let available = process.env.CACHEW_AVAILABLE || 'unknown';
  let cache = process.env.CACHEW_SELECTED_URL || '';
  if (cache && !config.caches.includes(cache)) cache = '';
  if (!config.valid || !python || fs.existsSync(disabled)) available = 'false';
  else if (available === 'true' && !cache) available = 'unknown';
  if (available === 'unknown') {
    cache = config.caches.find(candidate => healthy(candidate, config.healthTimeout)) || '';
    available = cache ? 'true' : 'false';
  }
  if (available !== 'true') fs.writeFileSync(disabled, '', {mode: 0o600});
  state.cache = available === 'true' ? cache : null;
  removeManaged(state, global);
  if (!fs.existsSync(disabled)) {
    const host = new URL(config.upstream).host;
    const entries = [[`url.${state.cache}/git/${host}/.insteadOf`, config.upstream + '/']];
    if (state.defaults[config.upstream]) entries.push(...authEntries(state.cache, state.defaults[config.upstream]));
    for (const [repository, auth] of Object.entries(state.tokens)) {
      if (!repository.startsWith(config.upstream + '/')) continue;
      const cached = `${state.cache}/git/${host}/${repository.slice(config.upstream.length + 1)}`;
      entries.push(...authEntries(cached, auth));
      if (!cached.endsWith('.git')) entries.push(...authEntries(cached + '.git', auth));
    }
    addManaged(state, global, entries);
  }
  state.active = !fs.existsSync(disabled);
  writeState(directory, state);
  if (python && state.active) {
    const bin = path.join(directory, 'bin');
    fs.mkdirSync(bin, {recursive: true});
    fs.writeFileSync(path.join(bin, 'git'), shim(state, directory, python, path.join(process.env.CACHEW_ACTION_PATH, 'git_wrapper.py')), {mode: 0o700});
    fs.appendFileSync(process.env.GITHUB_PATH, bin + '\n');
  }
  environment('GIT_CONFIG_GLOBAL', global);
  environment('CACHEW_GIT_FETCH_TIMEOUT', String(config.timeout));
  output('configured', String(state.active));
}

function cleanup(directory) {
  if (!directory || !fs.existsSync(directory)) return;
  const state = readState(directory);
  let restored = state.originalGlobal;
  if (restored === null) {
    // Actions can set, but cannot unset, a job environment variable. Restore
    // native HOME/XDG lookup for remaining posts through a secret-free proxy.
    restored = path.join(fs.realpathSync(process.env.RUNNER_TEMP), `cachew-original-${crypto.randomUUID()}.config`);
    writePrivate(restored, state.originalPaths.map(file => entry('include.path', file)).join(''));
  }
  environment('GIT_CONFIG_GLOBAL', restored);
  environment('CACHEW_SESSION_DIR', '');
  fs.rmSync(directory, {recursive: true, force: true});
}

if (require.main === module) {
  try {
    if (process.env['STATE_cachew-post']) cleanup(process.env['STATE_cachew-session']);
    else if (process.argv[2] === 'activate') activate();
    else prepareMain();
  } catch {
    if (process.argv[2] === 'activate') {
      const directory = process.env.CACHEW_SESSION_DIR;
      try {
        const state = readState(directory);
        disableInPlace(directory, state);
        writeState(directory, state);
        environment('GIT_CONFIG_GLOBAL', path.join(directory, 'global.config'));
      } catch {}
      output('configured', 'false');
      console.log('Cachew configuration unavailable; later Git operations use upstream.');
    } else if (process.env['STATE_cachew-post']) {
      // Post cleanup must not fail the job; RUNNER_TEMP is removed after the job.
      console.log('Cachew cleanup could not complete.');
    }
  }
}
