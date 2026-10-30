// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const cp = require('node:child_process');

const TRUSTED_CACHES = new Set([
  'https://cachew.dev.tenstorrent.net',
  'https://cachew.ci.tenstorrent.net',
  'https://cachew.yyz2.ci.tenstorrent.net',
]);
const PERFORMANCE = [
  ['checkout.workers', '0'],
  ['checkout.thresholdForParallelism', '50'],
  ['submodule.fetchJobs', '0'],
  ['gc.auto', '0'],
];

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

function origin(value) {
  const url = new URL(value);
  const loopback = url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname);
  if ((!loopback && url.protocol !== 'https:') || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
    throw new Error('Unsupported Git origin');
  }
  return url.origin;
}

function settings() {
  let upstream = 'https://github.com';
  let cache = null;
  let upstreamValid = true;
  let valid = true;
  try { upstream = origin(process.env.CACHEW_GITHUB_SERVER_URL || process.env.GITHUB_SERVER_URL); }
  catch { upstreamValid = false; valid = false; }
  try { cache = origin(process.env.CACHEW_CACHEW_URL); }
  catch { valid = false; }
  if (!cache || (!TRUSTED_CACHES.has(cache) && !(upstream.startsWith('http:') && cache.startsWith('http:')))) valid = false;
  const repository = process.env.CACHEW_REPOSITORY || process.env.GITHUB_REPOSITORY;
  const repositoryValid = /^[\w.-]+\/[\w.-]+$/.test(repository) && !repository.split('/').some(part => ['.', '..'].includes(part));
  if (!repositoryValid) valid = false;
  let timeout = Number(process.env.CACHEW_CACHEW_FETCH_TIMEOUT);
  let healthTimeout = Number(process.env.CACHEW_CACHEW_HEALTH_TIMEOUT);
  if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 600 || !Number.isFinite(healthTimeout) || healthTimeout <= 0 || healthTimeout > 30) {
    valid = false;
    timeout = 120;
    healthTimeout = 2;
  }
  return {upstream, cache, repository, timeout, healthTimeout, valid, upstreamValid, repositoryValid};
}

function git(state, args) {
  const env = {...process.env};
  for (const key of Object.keys(env)) if (key.startsWith('GIT_CONFIG_')) delete env[key];
  const result = cp.spawnSync(state.realGit, args, {env, encoding: 'utf8'});
  if (result.status !== 0) throw new Error('Unable to read Git configuration');
  return result.stdout;
}

function cacheTarget(value, selected) {
  try {
    const url = new URL(value);
    return url.pathname.startsWith('/git/') && (url.origin === selected || url.hostname === 'cachew' || url.hostname.startsWith('cachew.'));
  } catch { return false; }
}

function copyGlobals(directory, state, selectedCache, upstream, stripAuth) {
  if (state.originalGlobal === '') return '';
  const copies = new Map();
  let counter = 0;
  const originalHome = process.env.HOME || os.homedir();
  function copy(source) {
    source = path.resolve(source);
    if (copies.has(source)) return copies.get(source);
    if (!fs.existsSync(source)) return null;
    if (copies.size > 30) throw new Error('Too many included Git configuration files');
    const destination = path.join(directory, `${stripAuth ? 'clean' : 'base'}-${counter++}.config`);
    copies.set(source, destination);
    const contents = git(state, ['config', '--file', source, '--no-includes', '--null', '--list']);
    let result = '';
    for (const item of contents.split('\0')) {
      if (!item) continue;
      const separator = item.indexOf('\n');
      let key = separator < 0 ? item : item.slice(0, separator);
      let value = separator < 0 ? 'true' : item.slice(separator + 1);
      const lower = key.toLowerCase();
      if (lower.startsWith('url.') && lower.endsWith('.insteadof') && cacheTarget(key.slice(4, -10), selectedCache)) continue;
      if (lower.startsWith('http.') && lower.endsWith('.extraheader')) {
        const url = key.slice(5, -12);
        let isCache = false;
        let isUpstream = false;
        try {
          const parsed = new URL(url);
          isCache = parsed.origin === selectedCache || parsed.hostname === 'cachew' || parsed.hostname.startsWith('cachew.');
          isUpstream = parsed.origin === upstream;
        } catch {}
        if (isCache || (stripAuth && isUpstream)) continue;
      }
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

function prepare() {
  const config = settings();
  let directory = process.env.CACHEW_SESSION_DIR;
  let state;
  if (directory && fs.existsSync(directory)) state = readState(directory);
  else {
    directory = fs.mkdtempSync(path.join(fs.realpathSync(process.env.RUNNER_TEMP), 'cachew-'));
    fs.chmodSync(directory, 0o700);
    const originalHome = process.env.HOME || os.homedir();
    state = {version: 1, refs: 0, active: false, realGit: executable('git'), originalGlobal: process.env.GIT_CONFIG_GLOBAL ?? null, originalPaths: [path.join(process.env.XDG_CONFIG_HOME || path.join(originalHome, '.config'), 'git', 'config'), path.join(originalHome, '.gitconfig')], tokens: {}, defaults: {}, policies: {}};
    writeState(directory, state);
    save('cachew-session', directory);
    state.base = copyGlobals(directory, state, config.cache, config.upstream, false);
    state.clean = copyGlobals(directory, state, config.cache, config.upstream, true);
  }
  state.refs += 1;
  state.active = false;
  writePrivate(path.join(directory, 'global.config'), state.clean + PERFORMANCE.map(([key, value]) => entry(key, value)).join(''));
  if (!fs.existsSync(path.join(directory, 'upstream.config'))) fs.copyFileSync(path.join(directory, 'global.config'), path.join(directory, 'upstream.config'));
  writeState(directory, state);
  save('cachew-session', directory);
  output('session-dir', directory);
  environment('CACHEW_SESSION_DIR', directory);
  environment('CACHEW_REAL_GIT', state.realGit);
  environment('CACHEW_GIT_FETCH_TIMEOUT', String(config.timeout));
  environment('CACHEW_NODE', process.execPath);
  environment('CACHEW_CONFIG_HELPER', __filename);
  environment('GIT_CONFIG_GLOBAL', path.join(directory, 'global.config'));
}

function authEntry(url, header) {
  return entry(`http.${url}/.extraHeader`, '') + entry(`http.${url}/.extraHeader`, header);
}

function healthy(config) {
  const result = cp.spawnSync('curl', ['--disable', '--silent', '--fail', '--retry', '0', '--max-time', String(config.healthTimeout), '--output', os.devnull, '--write-out', '%{http_code}', config.cache + '/_liveness'], {timeout: (config.healthTimeout + 0.5) * 1000});
  return result.status === 0 && result.stdout?.toString() === '200';
}

function shellQuote(value) { return `'${value.replaceAll("'", "'\\''")}'`; }

function activate() {
  const directory = process.env.CACHEW_SESSION_DIR;
  const state = readState(directory);
  const config = settings();
  const token = process.env.CACHEW_TOKEN || '';
  const header = `AUTHORIZATION: basic ${Buffer.from(`x-access-token:${token}`).toString('base64')}`;
  mask(token);
  mask(header.slice('AUTHORIZATION: basic '.length));
  if (token && config.upstreamValid && config.repositoryValid && (process.env.CACHEW_PERSIST_CREDENTIALS || 'true').toLowerCase() !== 'false') {
    state.tokens[`${config.upstream}/${config.repository}`] = header;
    if (config.repository.toLowerCase() === process.env.GITHUB_REPOSITORY.toLowerCase()) state.defaults[config.upstream] = header;
  }
  let upstream = state.base + PERFORMANCE.map(([key, value]) => entry(key, value)).join('');
  for (const [url, auth] of Object.entries(state.defaults)) upstream += authEntry(url, auth);
  for (const [url, auth] of Object.entries(state.tokens)) {
    upstream += authEntry(url, auth);
    if (!url.endsWith('.git')) upstream += authEntry(url + '.git', auth);
  }
  writePrivate(path.join(directory, 'upstream.config'), upstream);
  let available = process.env.CACHEW_AVAILABLE || 'unknown';
  const disabled = path.join(directory, 'cache-disabled');
  let python;
  try {
    python = executable('python3');
    if (cp.spawnSync(python, ['-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)']).status !== 0) python = null;
  } catch {}
  if (!config.valid || !python || fs.existsSync(disabled)) available = 'false';
  else if (available === 'unknown') available = healthy(config) ? 'true' : 'false';
  if (available !== 'true') fs.writeFileSync(disabled, '', {mode: 0o600});
  else state.policies[config.upstream] = config.cache;
  let global = upstream;
  if (!fs.existsSync(disabled)) {
    for (const [url, cache] of Object.entries(state.policies)) {
      const host = new URL(url).host;
      global += entry(`url.${cache}/git/${host}/.insteadOf`, url + '/');
      if (state.defaults[url]) global += authEntry(cache, state.defaults[url]);
      for (const [repository, auth] of Object.entries(state.tokens)) {
        if (!repository.startsWith(url + '/')) continue;
        const cached = `${cache}/git/${host}/${repository.slice(url.length + 1)}`;
        global += authEntry(cached, auth);
        if (!cached.endsWith('.git')) global += authEntry(cached + '.git', auth);
      }
    }
  }
  state.active = !fs.existsSync(disabled);
  writePrivate(path.join(directory, 'global.config'), global);
  writeState(directory, state);
  if (python) {
    const bin = path.join(directory, 'bin');
    fs.mkdirSync(bin, {recursive: true});
    const wrapper = path.join(process.env.CACHEW_ACTION_PATH, 'git_wrapper.py');
    fs.writeFileSync(path.join(bin, 'git'), `#!/bin/sh\nexport CACHEW_REAL_GIT=${shellQuote(state.realGit)}\nexport CACHEW_SESSION_DIR=${shellQuote(directory)}\nexec ${shellQuote(python)} ${shellQuote(wrapper)} "$@"\n`, {mode: 0o700});
    fs.appendFileSync(process.env.GITHUB_PATH, bin + '\n');
  }
  output('configured', String(state.active));
}

function cleanup(directory) {
  if (!directory || !fs.existsSync(directory)) return;
  const state = readState(directory);
  state.refs -= 1;
  if (state.refs <= 0) {
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
  else writeState(directory, state);
}

try {
  if (process.env['STATE_cachew-session']) cleanup(process.env['STATE_cachew-session']);
  else if (process.argv[2] === 'activate') activate();
  else prepare();
} catch {
  if (process.argv[2] === 'activate') {
    const directory = process.env.CACHEW_SESSION_DIR;
    try {
      readState(directory);
      fs.writeFileSync(path.join(directory, 'cache-disabled'), '', {mode: 0o600});
      writePrivate(path.join(directory, 'global.config'), fs.readFileSync(path.join(directory, 'upstream.config')));
    } catch {}
    output('configured', 'false');
    console.log('Cachew configuration unavailable; later Git operations use upstream.');
  } else {
    console.error('Unable to isolate or clean up Git configuration.');
    process.exitCode = 1;
  }
}
